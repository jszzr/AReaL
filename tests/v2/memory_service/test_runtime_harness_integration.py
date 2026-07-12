# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from examples.memory_service import runtime_codebook_adapter as runtime_adapter
from examples.memory_service import scoped_codebook_eval as harness

from areal.v2.memory_service import (
    InMemoryMemoryRuntimeStore,
    MemoryExposureStatus,
    MemoryQuerySpecV1,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

_TASK_POLICY_VERSION = hashlib.sha256(b"frozen-scripted-policy-v1").hexdigest()
_REFERENCE_HEADER = (
    b"[memory-codebook/v1]\n[mask=XXXXX means unavailable; answer UNKNOWN]\n"
)


def _query(case: harness.CodebookCase) -> bytes:
    return (
        f"What is the current code for {case.target_key}? "
        "Reply with exactly the code or UNKNOWN."
    ).encode()


def _release_id(references, arm: str) -> str:
    return {
        "current_release": references.releases.current_release_id,
        "memory_off": references.releases.empty_release_id,
        "target_masked": references.releases.masked_release_id,
        "stale_release": references.releases.stale_release_id,
    }[arm]


def _runtime(source: SQLiteMemoryStore) -> InMemoryMemoryRuntimeStore:
    return InMemoryMemoryRuntimeStore(
        source,
        source,
        retrievers=(runtime_adapter.ReleaseManifestRetrieverV1(),),
        renderers=(runtime_adapter.CodebookReleaseRendererV1(),),
        consumers=(runtime_adapter.ScriptedCodebookConsumerV1(),),
    )


def _execute_runtime(
    source: SQLiteMemoryStore,
    assignment: harness.ReleaseSourceAssignment,
    query: bytes,
    *,
    future_run_id: str,
):
    runtime = _runtime(source)
    spec = MemoryQuerySpecV1(
        scope=assignment.scope,
        release_id=assignment.release_id,
        trajectory_id=future_run_id,
        rollout_group_id="frozen-six-arm-group",
        query_sequence_no=0,
        query_sha256=hashlib.sha256(query).hexdigest(),
        task_policy_id="frozen-scripted-agent",
        task_policy_version_sha256=_TASK_POLICY_VERSION,
        retrieval_policy_id=(runtime_adapter.RELEASE_ORDER_RETRIEVER_ID_V1),
        retrieval_policy_version_sha256=(
            runtime_adapter.RELEASE_ORDER_RETRIEVER_VERSION_SHA256_V1
        ),
        max_returned_items=16,
        max_context_utf8_bytes=4096,
        idempotency_key=f"runtime-{future_run_id}",
    )
    attempt = runtime.begin_query(spec)
    result = runtime.resolve_query(
        assignment.scope,
        attempt.attempt_id,
        query=query,
    )
    delivery = runtime.prepare_delivery(
        assignment.scope,
        result.query_result_id,
        renderer_id=runtime_adapter.CODEBOOK_RENDERER_ID_V1,
        renderer_version_sha256=(runtime_adapter.CODEBOOK_RENDERER_VERSION_SHA256_V1),
    )
    exposure, response = runtime.submit_delivery(
        assignment.scope,
        delivery.delivery_id,
        consumer_id=runtime_adapter.SCRIPTED_CONSUMER_ID_V1,
        consumer_version_sha256=(runtime_adapter.SCRIPTED_CONSUMER_VERSION_SHA256_V1),
        call_id=future_run_id,
        query=query,
        history=(),
    )
    ack = runtime.get_consumer_ack(
        assignment.scope,
        exposure.consumer_ack_id,
    )
    return runtime, result, delivery, ack, exposure, response


def _reference_render(result) -> harness.RenderedContext:
    """Independent tiny renderer used only by the evaluator side of the test."""

    if not result.returned_items:
        return harness.RenderedContext(bytes=b"", entry_receipts=())
    chunks = [_REFERENCE_HEADER]
    receipts = []
    offset = len(_REFERENCE_HEADER)
    for item in result.returned_items:
        key, separator, value = item.content.partition(" = ")
        if separator != " = ":
            raise AssertionError("runtime item does not use the frozen fact grammar")
        line = f"{item.release_position:02d}\t{key}\t{value}\n".encode("ascii")
        end = offset + len(line)
        receipts.append(
            harness.EntryReceipt(
                slot=item.release_position,
                key=key,
                value=value,
                source_kind="release",
                revision_id=item.revision.revision_id,
                candidate_id=item.candidate_id,
                evidence_ids=tuple(evidence.evidence_id for evidence in item.evidence),
                content_sha256=hashlib.sha256(f"{key}\t{value}".encode()).hexdigest(),
                rendered_start=offset,
                rendered_end=end,
            )
        )
        chunks.append(line)
        offset = end
    return harness.RenderedContext(
        bytes=b"".join(chunks),
        entry_receipts=tuple(receipts),
    )


def _runtime_observation(
    *,
    references,
    assignment,
    future_run_id,
    query,
    result,
    delivery,
    ack,
    exposure,
    response,
    audit,
) -> harness.ExecutionObservation:
    rendered = _reference_render(result)
    assert (
        delivery.rendered_context_sha256 == hashlib.sha256(rendered.bytes).hexdigest()
    )
    assert delivery.rendered_context_utf8_bytes == len(rendered.bytes)
    assert tuple(
        (
            item.revision.revision_id,
            item.rendered_start,
            item.rendered_end,
        )
        for item in delivery.rendered_spans
    ) == tuple(
        (item.revision_id, item.rendered_start, item.rendered_end)
        for item in rendered.entry_receipts
    )
    assert (
        ack.submitted_prompt_context_sha256
        == hashlib.sha256(rendered.bytes).hexdigest()
    )
    assert ack.submitted_prompt_context_utf8_bytes == len(rendered.bytes)

    runtime_treatment = harness.ResolvedTreatment(
        source_kind="release",
        scope=assignment.scope,
        release_id=assignment.release_id,
        eligible_ids=tuple(item.revision_id for item in exposure.eligible_revisions),
        retrieved_ids=tuple(item.revision_id for item in exposure.retrieved_revisions),
        returned_ids=tuple(item.revision_id for item in exposure.returned_revisions),
        source_evidence_ids=tuple(
            evidence.evidence_id
            for item in result.returned_items
            for evidence in item.evidence
        ),
        entries=(),
    )
    consumer_result = harness.ConsumerResult(
        response=response,
        input_receipt=harness.ConsumerInputReceipt(
            received_context_sha256=ack.submitted_prompt_context_sha256,
            received_context_utf8_bytes=(ack.submitted_prompt_context_utf8_bytes),
            received_query_sha256=ack.observed_query_sha256,
            received_history_length=ack.observed_history_length,
        ),
    )
    return harness.make_execution_observation(
        execution_index=0,
        treatment=runtime_treatment,
        reader_audit=audit,
        rendered_context=rendered,
        query=query,
        consumer_result=consumer_result,
        model_call_receipt=None,
        capture_session_ids=references.capture.capture_session_ids,
        future_session_id=f"{future_run_id}-session",
        future_run_id=future_run_id,
        capture_pid=101,
        future_pid=202,
        capture_process_instance_id="capture-process",
        future_process_instance_id="future-process",
    )


@pytest.mark.parametrize(
    ("arm", "status"),
    (
        ("current_release", MemoryExposureStatus.DELIVERED),
        ("memory_off", MemoryExposureStatus.MEMORY_OFF),
        ("target_masked", MemoryExposureStatus.DELIVERED),
        ("stale_release", MemoryExposureStatus.DELIVERED),
    ),
)
def test_runtime_executes_before_private_parent_validation(
    tmp_path: Path,
    arm: str,
    status: MemoryExposureStatus,
) -> None:
    case_index = {
        "current_release": 0,
        "memory_off": 1,
        "target_masked": 2,
        "stale_release": 3,
    }[arm]
    case = harness.generate_case(case_index)
    database_path = tmp_path / f"{arm}.sqlite3"
    references = harness.build_case_database(case, database_path)
    assignment = harness.ReleaseSourceAssignment(
        scope=references.capture.local_scope,
        release_id=_release_id(references, arm),
    )
    query = _query(case)
    future_run_id = f"{case.case_id}-runtime-{arm}"

    # Runtime runs with only its scoped release assignment and query.  No
    # evaluator trace, expected response, or selected IDs are fed into it.
    source = SQLiteMemoryStore(database_path)
    runtime, result, delivery, ack, exposure, response = _execute_runtime(
        source,
        assignment,
        query,
        future_run_id=future_run_id,
    )
    assert exposure.status is status
    assert runtime.list_exposures(assignment.scope) == (exposure,)

    # Only after the immutable runtime record exists does an independent
    # evaluator reopen the source, reconstruct provenance, and see case truth.
    # Its audit below is explicitly reconstruction evidence for reusing the
    # frozen scorer; it is not claimed to be a runtime source-read receipt.
    evaluator_source = SQLiteMemoryStore(database_path)
    audit = harness.ReadAuditSink()
    capability = harness.ReleaseReadCapability(
        evaluator_source,
        assignment,
        audit,
    )
    expected_treatment = harness.resolve_treatment(capability)
    harness.validate_resolved_treatment(
        database_path,
        assignment,
        expected_treatment,
        audit.snapshot(),
    )
    observation = _runtime_observation(
        references=references,
        assignment=assignment,
        future_run_id=future_run_id,
        query=query,
        result=result,
        delivery=delivery,
        ack=ack,
        exposure=exposure,
        response=response,
        audit=audit.snapshot(),
    )
    assert observation.eligible_ids == expected_treatment.eligible_ids
    assert observation.retrieved_ids == expected_treatment.retrieved_ids
    assert observation.returned_ids == expected_treatment.returned_ids
    assert observation.source_evidence_ids == (expected_treatment.source_evidence_ids)

    schedule = harness.make_parent_schedule_item(
        execution_index=0,
        case=case,
        references=references,
        arm=arm,
    )
    (trace,) = harness.parent_join_and_score(
        (observation,),
        (schedule,),
        enforce_scripted_outcomes=True,
    )
    assert trace.response == response
    assert trace.expected_response == schedule.expected_response
    assert trace.injected_revision_ids == tuple(
        item.revision_id for item in exposure.injected_revisions
    )


def test_parent_rejects_context_receipt_drift_after_runtime_execution(
    tmp_path: Path,
) -> None:
    case = harness.generate_case(0)
    database_path = tmp_path / "receipt-drift.sqlite3"
    references = harness.build_case_database(case, database_path)
    assignment = harness.ReleaseSourceAssignment(
        scope=references.capture.local_scope,
        release_id=references.releases.current_release_id,
    )
    query = _query(case)
    future_run_id = f"{case.case_id}-runtime-drift"
    source = SQLiteMemoryStore(database_path)
    _runtime_store, result, delivery, ack, exposure, response = _execute_runtime(
        source,
        assignment,
        query,
        future_run_id=future_run_id,
    )

    audit = harness.ReadAuditSink()
    capability = harness.ReleaseReadCapability(
        SQLiteMemoryStore(database_path), assignment, audit
    )
    harness.resolve_treatment(capability)
    observation = _runtime_observation(
        references=references,
        assignment=assignment,
        future_run_id=future_run_id,
        query=query,
        result=result,
        delivery=delivery,
        ack=ack,
        exposure=exposure,
        response=response,
        audit=audit.snapshot(),
    )
    forged_hash = "0" * 64
    forged = replace(
        observation,
        rendered_context_sha256=forged_hash,
        consumer_input_receipt=replace(
            observation.consumer_input_receipt,
            received_context_sha256=forged_hash,
        ),
    )
    schedule = harness.make_parent_schedule_item(
        execution_index=0,
        case=case,
        references=references,
        arm="current_release",
    )

    with pytest.raises(harness.ObservationValidationError):
        harness.parent_join_and_score(
            (forged,),
            (schedule,),
            enforce_scripted_outcomes=True,
        )
