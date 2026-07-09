# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic Memory Service helpfulness evaluator."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from examples.memory_service import scoped_codebook_eval as helpfulness

from areal.v2.memory_service import (
    EvidenceEvent,
    EvidenceKind,
    MemoryScope,
    RevisionOperation,
)
from areal.v2.memory_service.errors import ReleaseNotFoundError
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

GOLDEN_CASE_ZERO = (
    b'{"case_id":"nonce-000","case_index":0,"current_value":"GSPFA",'
    b'"masked_value":"XXXXX","old_value":"PYZLF","padding_entry":'
    b'{"key":"project-btggtx","value":"Q9W8Z"},"schema_version":1,'
    b'"seed":"areal-memory-helpfulness-v1-20260708","shared_entries":'
    b'[{"key":"project-4zd9zq","value":"DWBQX"},{"key":"project-yrmszc",'
    b'"value":"M953M"},{"key":"project-hbh7xn","value":"4SSR4"},'
    b'{"key":"project-arz5e4","value":"86HWQ"}],"subject_id":'
    b'"nonce-subject-000","target_key":"project-xk527d","target_slot":0}'
)
GOLDEN_CASE_ZERO_SHA256 = (
    "5bfc563a0be801da69e482cd4c4caaa5d0d9e4f0176527391229209282e21f95"
)
RENDER_HEADER = (
    b"[memory-codebook/v1]\n[mask=XXXXX means unavailable; answer UNKNOWN]\n"
)
FIRST_EIGHT_MANIFEST_SHA256 = (
    "5bfc563a0be801da69e482cd4c4caaa5d0d9e4f0176527391229209282e21f95",
    "68406e7569543e4f94beb255074a664580604fbc8a6018dde74ce1b9e86a7ed0",
    "bcdf689a9f8f74c2d802a9163f06298d5db28df1c071c6b85a421d364fa04eaf",
    "e534a9a1de05c020dc0cfb1ca673b11b07c36577aa86b77d95a715a85fefc9bf",
    "cd788dbda47cba5d0bebd170e8b2d87776ce5ccdaf2147228180bbd22f2ccc8a",
    "db2365cc5d809576bba3c3940e9dead142949e7d87122ec715b2b96d7c3d6b97",
    "f9ca0208b6a6b485404b39a66d9215d95d8c566b9d0124514aae5cbac9b222f7",
    "1f6c8caa43ffe41463613b6601ef4c4796bdef03b868666a1a06d449d9236ea1",
)


def test_helpfulness_module_imports() -> None:
    module = importlib.import_module("examples.memory_service.scoped_codebook_eval")

    assert module.__name__ == "examples.memory_service.scoped_codebook_eval"


def _core_entries(
    case: helpfulness.CodebookCase,
    *,
    target_value: str,
    source_kind: str,
) -> tuple[helpfulness.ResolvedEntry, ...]:
    shared_slots = iter(slot for slot in range(5) if slot != case.target_slot)
    entries = [
        helpfulness.ResolvedEntry(
            slot=case.target_slot,
            key=case.target_key,
            value=target_value,
            source_kind=source_kind,
            revision_id="revision-target" if source_kind == "release" else None,
            candidate_id="candidate-target" if source_kind == "release" else None,
            evidence_ids=() if source_kind == "oracle" else ("evidence-target",),
        )
    ]
    for shared in case.shared_entries:
        entries.append(
            helpfulness.ResolvedEntry(
                slot=next(shared_slots),
                key=shared.key,
                value=shared.value,
                source_kind=source_kind,
                revision_id=(
                    f"revision-{shared.key}" if source_kind == "release" else None
                ),
                candidate_id=(
                    f"candidate-{shared.key}" if source_kind == "release" else None
                ),
                evidence_ids=()
                if source_kind == "oracle"
                else (f"evidence-{shared.key}",),
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.slot))


def _case_entries(
    case: helpfulness.CodebookCase,
    *,
    target_value: str,
    source_kind: str,
) -> tuple[helpfulness.ResolvedEntry, ...]:
    core = _core_entries(
        case,
        target_value=target_value,
        source_kind=source_kind,
    )
    padding = helpfulness.ResolvedEntry(
        slot=5,
        key=case.padding_entry.key,
        value=case.padding_entry.value,
        source_kind=source_kind,
        revision_id="revision-padding" if source_kind == "release" else None,
        candidate_id="candidate-padding" if source_kind == "release" else None,
        evidence_ids=() if source_kind == "oracle" else ("evidence-padding",),
    )
    return (*core, padding)


def _raw_entries(
    case: helpfulness.CodebookCase,
) -> tuple[helpfulness.ResolvedEntry, ...]:
    old_and_shared = _core_entries(
        case,
        target_value=case.old_value,
        source_kind="raw_evidence",
    )
    correction = helpfulness.ResolvedEntry(
        slot=case.target_slot,
        key=case.target_key,
        value=case.current_value,
        source_kind="raw_evidence",
        evidence_ids=("evidence-current",),
    )
    return (*old_and_shared, correction)


def _query(case: helpfulness.CodebookCase) -> bytes:
    return (
        f"What is the current code for {case.target_key}? "
        "Reply with exactly the code or UNKNOWN."
    ).encode()


def test_case_zero_matches_literal_golden_json_and_sha256() -> None:
    case = helpfulness.generate_case(0)

    manifest = helpfulness.case_manifest_bytes(case)

    assert manifest == GOLDEN_CASE_ZERO
    assert sha256(manifest).hexdigest() == GOLDEN_CASE_ZERO_SHA256
    assert helpfulness.case_manifest_sha256(case) == GOLDEN_CASE_ZERO_SHA256


def test_first_eight_cases_are_deterministic_unique_and_slot_balanced() -> None:
    cases = tuple(helpfulness.generate_case(index) for index in range(8))

    assert tuple(helpfulness.case_manifest_sha256(case) for case in cases) == (
        FIRST_EIGHT_MANIFEST_SHA256
    )
    assert len({case.case_id for case in cases}) == 8
    assert len({helpfulness.case_manifest_sha256(case) for case in cases}) == 8
    assert Counter(case.target_slot for case in cases) == {
        0: 2,
        1: 2,
        2: 2,
        3: 1,
        4: 1,
    }
    for index, case in enumerate(cases):
        keys = [
            case.target_key,
            *(entry.key for entry in case.shared_entries),
            case.padding_entry.key,
        ]
        values = [
            case.old_value,
            case.current_value,
            *(entry.value for entry in case.shared_entries),
            case.padding_entry.value,
        ]
        assert case.case_index == index
        assert case.target_slot == index % 5
        assert len(keys) == len(set(keys))
        assert len(values) == len(set(values))
        assert not {"UNKNOWN", "XXXXX"}.intersection(values)


def test_collision_attempt_changes_only_the_colliding_field(monkeypatch) -> None:
    baseline = helpfulness.generate_case(1)
    real_token = helpfulness._token
    attempts: list[tuple[str, int, int]] = []

    def collide_current_once(
        seed: str,
        case_index: int,
        label: str,
        item_index: int,
        attempt: int,
        length: int,
    ) -> str:
        attempts.append((label, item_index, attempt))
        if label == "current" and attempt == 0:
            return baseline.old_value
        return real_token(seed, case_index, label, item_index, attempt, length)

    monkeypatch.setattr(helpfulness, "_token", collide_current_once)

    collided = helpfulness.generate_case(1)
    baseline_fields = asdict(baseline)
    collided_fields = asdict(collided)
    changed_fields = {
        name
        for name, value in baseline_fields.items()
        if value != collided_fields[name]
    }

    assert changed_fields == {"current_value"}
    assert collided.current_value not in {
        collided.old_value,
        "UNKNOWN",
        collided.masked_value,
    }
    assert ("current", 0, 0) in attempts
    assert ("current", 0, 1) in attempts
    assert all(
        attempt == 0 for label, _item_index, attempt in attempts if label != "current"
    )


def test_renderer_emits_exact_current_masked_raw_and_empty_bytes() -> None:
    case = helpfulness.generate_case(0)
    current = helpfulness.render_context(
        _case_entries(case, target_value=case.current_value, source_kind="release")
    )
    masked = helpfulness.render_context(
        _case_entries(case, target_value=case.masked_value, source_kind="release")
    )
    raw = helpfulness.render_context(_raw_entries(case))
    empty = helpfulness.render_context(())

    assert current.bytes == RENDER_HEADER + (
        b"00\tproject-xk527d\tGSPFA\n"
        b"01\tproject-4zd9zq\tDWBQX\n"
        b"02\tproject-yrmszc\tM953M\n"
        b"03\tproject-hbh7xn\t4SSR4\n"
        b"04\tproject-arz5e4\t86HWQ\n"
        b"05\tproject-btggtx\tQ9W8Z\n"
    )
    assert masked.bytes == RENDER_HEADER + (
        b"00\tproject-xk527d\tXXXXX\n"
        b"01\tproject-4zd9zq\tDWBQX\n"
        b"02\tproject-yrmszc\tM953M\n"
        b"03\tproject-hbh7xn\t4SSR4\n"
        b"04\tproject-arz5e4\t86HWQ\n"
        b"05\tproject-btggtx\tQ9W8Z\n"
    )
    assert raw.bytes == RENDER_HEADER + (
        b"00\tproject-xk527d\tPYZLF\n"
        b"01\tproject-4zd9zq\tDWBQX\n"
        b"02\tproject-yrmszc\tM953M\n"
        b"03\tproject-hbh7xn\t4SSR4\n"
        b"04\tproject-arz5e4\t86HWQ\n"
        b"00\tproject-xk527d\tGSPFA\n"
    )
    assert empty.bytes == b""
    assert empty.entry_receipts == ()
    assert (
        helpfulness.consume_scripted(_query(case), masked.bytes).response == "UNKNOWN"
    )


def test_renderer_receipts_use_full_line_half_open_ranges() -> None:
    case = helpfulness.generate_case(0)
    rendered = helpfulness.render_context(_raw_entries(case))

    assert rendered.entry_receipts[0].rendered_start == len(RENDER_HEADER)
    assert rendered.entry_receipts[-1].rendered_end == len(rendered.bytes)
    for receipt in rendered.entry_receipts:
        rendered_line = rendered.bytes[receipt.rendered_start : receipt.rendered_end]
        assert rendered_line == (
            f"{receipt.slot:02d}\t{receipt.key}\t{receipt.value}\n".encode()
        )
        assert rendered_line.endswith(b"\n")
        assert (
            receipt.content_sha256
            == sha256(f"{receipt.key}\t{receipt.value}".encode()).hexdigest()
        )


def test_current_and_oracle_render_byte_identically() -> None:
    case = helpfulness.generate_case(3)
    current = helpfulness.render_context(
        _case_entries(case, target_value=case.current_value, source_kind="release")
    )
    oracle = helpfulness.render_context(
        _case_entries(case, target_value=case.current_value, source_kind="oracle")
    )

    assert current.bytes == oracle.bytes
    assert sha256(current.bytes).digest() == sha256(oracle.bytes).digest()
    assert (
        tuple(receipt.source_kind for receipt in current.entry_receipts)
        == ("release",) * 6
    )
    assert (
        tuple(receipt.source_kind for receipt in oracle.entry_receipts)
        == ("oracle",) * 6
    )


def test_scripted_consumer_uses_last_raw_occurrence_and_returns_input_receipt() -> None:
    case = helpfulness.generate_case(0)
    rendered = helpfulness.render_context(_raw_entries(case))
    query = _query(case)

    result = helpfulness.consume_scripted(query, rendered.bytes)

    assert result.response == case.current_value
    assert (
        result.input_receipt.received_context_sha256
        == sha256(rendered.bytes).hexdigest()
    )
    assert result.input_receipt.received_context_utf8_bytes == len(rendered.bytes)
    assert result.input_receipt.received_query_sha256 == sha256(query).hexdigest()
    assert helpfulness.consume_scripted(query, b"").response == "UNKNOWN"


def test_normalization_accepts_only_exact_nonce_or_unknown() -> None:
    assert helpfulness.parse_fact(
        "project-xk527d = GSPFA"
    ) == helpfulness.CodebookEntry(
        key="project-xk527d",
        value="GSPFA",
    )
    for malformed in (
        " project-xk527d = GSPFA",
        "project-xk527d=GSPFA",
        "project-xk527d = GSPFA ",
        "project-xk527d = GSPFA = EXTRA",
    ):
        with pytest.raises(ValueError, match="fact"):
            helpfulness.parse_fact(malformed)

    assert helpfulness.normalize_response("  gspfa  ") == "GSPFA"
    assert helpfulness.normalize_response("ＧＳＰＦＡ") == "GSPFA"
    assert helpfulness.normalize_response(" unknown\n") == "UNKNOWN"
    assert helpfulness.normalize_response("GSPFA is the code") == "GSPFA IS THE CODE"
    assert helpfulness.normalize_response("XXXXX") == "XXXXX"


def test_abstained_and_utility_are_exact() -> None:
    assert helpfulness.abstained("UNKNOWN") is True
    assert helpfulness.abstained("GSPFA") is False
    assert helpfulness.utility("GSPFA", current_value="GSPFA") == 1
    assert helpfulness.utility("UNKNOWN", current_value="GSPFA") == 0
    assert helpfulness.utility("XXXXX", current_value="GSPFA") == -1
    assert helpfulness.utility("PYZLF", current_value="GSPFA") == -1
    assert helpfulness.utility("GSPFA EXTRA", current_value="GSPFA") == -1


def _build_case_graph(
    tmp_path: Path,
    *,
    case_index: int = 2,
) -> tuple[
    helpfulness.CodebookCase,
    Path,
    helpfulness.CaseDatabaseReferences,
    SQLiteMemoryStore,
]:
    case = helpfulness.generate_case(case_index)
    database_path = tmp_path / f"{case.case_id}.sqlite3"
    references = helpfulness.build_case_database(case, database_path)
    return case, database_path, references, SQLiteMemoryStore(database_path)


def _local_revision_ids(
    references: helpfulness.CaseDatabaseReferences,
) -> tuple[str, ...]:
    revisions = references.revisions
    return (
        revisions.target_old_revision_id,
        revisions.target_current_revision_id,
        *revisions.shared_revision_ids,
        revisions.padding_revision_id,
        revisions.target_masked_revision_id,
    )


def _expected_release_payloads(
    case: helpfulness.CodebookCase,
    *,
    target_value: str,
) -> tuple[str, ...]:
    shared = iter(case.shared_entries)
    return tuple(
        f"{case.target_key} = {target_value}"
        if slot == case.target_slot
        else f"{(entry := next(shared)).key} = {entry.value}"
        for slot in range(5)
    ) + (f"{case.padding_entry.key} = {case.padding_entry.value}",)


def test_capture_builds_exact_local_and_foreign_graph(tmp_path: Path) -> None:
    case, _path, references, store = _build_case_graph(tmp_path)
    capture = references.capture

    local_records = store.list(capture.local_scope)
    foreign_records = store.list(capture.foreign_scope)

    assert capture.local_scope == MemoryScope(
        "memory-eval",
        "scoped-codebook-v1",
        case.subject_id,
    )
    assert capture.foreign_scope == MemoryScope(
        "memory-eval",
        "scoped-codebook-v1",
        f"{case.subject_id}-foreign",
    )
    assert len(local_records) == 8
    assert Counter(record.event.kind for record in local_records) == {
        EvidenceKind.USER_MESSAGE: 5,
        EvidenceKind.FEEDBACK: 1,
        EvidenceKind.ENVIRONMENT: 2,
    }
    assert len(foreign_records) == 1
    assert foreign_records[0].event.kind is EvidenceKind.ENVIRONMENT
    assert (
        foreign_records[0].event.payload == f"{case.target_key} = {case.current_value}"
    )
    assert len(store.list_candidates(capture.local_scope)) == 8
    assert len(store.list_revisions(capture.local_scope)) == 8
    assert len(store.list_releases(capture.local_scope)) == 4
    assert len(store.list_candidates(capture.foreign_scope)) == 1
    assert len(store.list_revisions(capture.foreign_scope)) == 1
    assert len(store.list_releases(capture.foreign_scope)) == 1

    for revision_id in _local_revision_ids(references):
        revision = store.get_revision(capture.local_scope, revision_id)
        candidate = store.get_candidate(
            capture.local_scope,
            revision.proposal.candidate_id,
        )
        assert candidate.proposal.evidence_ids
        assert all(
            store.get(capture.local_scope, evidence_id).event.scope
            == capture.local_scope
            for evidence_id in candidate.proposal.evidence_ids
        )
    foreign_revision = store.get_revision(
        capture.foreign_scope,
        references.revisions.foreign_target_revision_id,
    )
    foreign_candidate = store.get_candidate(
        capture.foreign_scope,
        foreign_revision.proposal.candidate_id,
    )
    assert foreign_candidate.proposal.evidence_ids == (capture.foreign_evidence_id,)


def test_capture_records_store_frozen_kinds_timestamps_and_cutoff_boundary(
    tmp_path: Path,
) -> None:
    case, _path, references, store = _build_case_graph(tmp_path, case_index=3)
    capture = references.capture
    case_base = datetime(2026, 7, 8, tzinfo=UTC) + timedelta(days=case.case_index)

    assert capture.case_base == case_base
    assert capture.raw_history_cutoff == case_base + timedelta(seconds=90)
    assert capture.capture_session_ids == (
        "nonce-003-capture-old",
        "nonce-003-capture-new",
        "nonce-003-capture-control",
    )

    old_records = store.list(
        capture.local_scope,
        session_id=capture.capture_session_ids[0],
    )
    new_records = store.list(
        capture.local_scope,
        session_id=capture.capture_session_ids[1],
    )
    control_records = store.list(
        capture.local_scope,
        session_id=capture.capture_session_ids[2],
    )
    expected_old = _core_entries(
        case,
        target_value=case.old_value,
        source_kind="raw_evidence",
    )

    assert tuple(record.event.sequence_no for record in old_records) == tuple(range(5))
    assert tuple(record.event.observed_at for record in old_records) == tuple(
        case_base + timedelta(seconds=slot) for slot in range(5)
    )
    assert tuple(record.event.payload for record in old_records) == tuple(
        f"{entry.key} = {entry.value}" for entry in expected_old
    )
    assert (
        tuple(record.event.session_id for record in old_records)
        == ("nonce-003-capture-old",) * 5
    )
    assert (
        tuple(record.event.run_id for record in old_records)
        == ("nonce-003-run-old",) * 5
    )
    assert tuple(record.event.idempotency_key for record in old_records) == tuple(
        f"nonce-003-evidence-old-{slot:02d}" for slot in range(5)
    )
    assert capture.old_evidence_ids == tuple(
        record.evidence_id for record in old_records
    )
    assert all(record.event.kind is EvidenceKind.USER_MESSAGE for record in old_records)
    assert len(new_records) == 1
    assert new_records[0].event.kind is EvidenceKind.FEEDBACK
    assert new_records[0].event.sequence_no == 0
    assert new_records[0].event.observed_at == case_base + timedelta(seconds=60)
    assert new_records[0].event.observed_at <= capture.raw_history_cutoff
    assert new_records[0].event.payload == f"{case.target_key} = {case.current_value}"
    assert new_records[0].event.session_id == "nonce-003-capture-new"
    assert new_records[0].event.run_id == "nonce-003-run-new"
    assert (
        new_records[0].event.idempotency_key
        == f"nonce-003-evidence-new-{case.target_slot:02d}"
    )
    assert capture.current_evidence_id == new_records[0].evidence_id
    assert tuple(record.event.kind for record in control_records) == (
        EvidenceKind.ENVIRONMENT,
        EvidenceKind.ENVIRONMENT,
    )
    assert tuple(record.event.sequence_no for record in control_records) == (0, 1)
    assert tuple(record.event.observed_at for record in control_records) == (
        case_base + timedelta(seconds=120),
        case_base + timedelta(seconds=121),
    )
    assert tuple(record.event.payload for record in control_records) == (
        f"{case.padding_entry.key} = {case.padding_entry.value}",
        f"{case.target_key} = {case.masked_value}",
    )
    assert (
        tuple(record.event.session_id for record in control_records)
        == ("nonce-003-capture-control",) * 2
    )
    assert (
        tuple(record.event.run_id for record in control_records)
        == ("nonce-003-run-control",) * 2
    )
    assert tuple(record.event.idempotency_key for record in control_records) == (
        "nonce-003-evidence-control-05",
        f"nonce-003-evidence-control-{case.target_slot:02d}",
    )
    assert capture.control_evidence_ids == tuple(
        record.evidence_id for record in control_records
    )
    assert all(
        record.event.observed_at > capture.raw_history_cutoff
        for record in control_records
    )
    foreign = store.get(capture.foreign_scope, capture.foreign_evidence_id)
    assert foreign.event.observed_at == case_base + timedelta(seconds=180)
    assert foreign.event.session_id == "nonce-003-capture-foreign"
    assert foreign.event.run_id == "nonce-003-run-foreign"
    assert foreign.event.sequence_no == 0
    assert (
        foreign.event.idempotency_key
        == f"nonce-003-evidence-foreign-target-current-{case.target_slot:02d}"
    )


def test_candidate_and_revision_idempotency_keys_are_exact(tmp_path: Path) -> None:
    case, _path, references, store = _build_case_graph(tmp_path)
    capture = references.capture
    revisions = references.revisions
    shared_slots = tuple(slot for slot in range(5) if slot != case.target_slot)
    local_rows = (
        (
            "target-old",
            case.target_slot,
            revisions.target_old_revision_id,
            capture.old_evidence_ids[case.target_slot],
        ),
        (
            "target-current",
            case.target_slot,
            revisions.target_current_revision_id,
            capture.current_evidence_id,
        ),
        *tuple(
            (
                "shared",
                slot,
                revision_id,
                capture.old_evidence_ids[slot],
            )
            for slot, revision_id in zip(
                shared_slots,
                revisions.shared_revision_ids,
                strict=True,
            )
        ),
        (
            "padding",
            5,
            revisions.padding_revision_id,
            capture.control_evidence_ids[0],
        ),
        (
            "target-masked",
            case.target_slot,
            revisions.target_masked_revision_id,
            capture.control_evidence_ids[1],
        ),
    )

    assert (
        "case"
        not in inspect.signature(helpfulness._append_grounded_revision).parameters
    )
    assert (
        "case_id" in inspect.signature(helpfulness._append_grounded_revision).parameters
    )
    for role, slot, revision_id, evidence_id in local_rows:
        revision = store.get_revision(capture.local_scope, revision_id)
        candidate = store.get_candidate(
            capture.local_scope,
            revision.proposal.candidate_id,
        )
        assert candidate.proposal.idempotency_key == (
            f"{case.case_id}-candidate-local-{role}-{slot:02d}"
        )
        assert revision.proposal.idempotency_key == (
            f"{case.case_id}-revision-local-{role}-{slot:02d}"
        )
        assert candidate.proposal.evidence_ids == (evidence_id,)

    foreign_revision = store.get_revision(
        capture.foreign_scope,
        revisions.foreign_target_revision_id,
    )
    foreign_candidate = store.get_candidate(
        capture.foreign_scope,
        foreign_revision.proposal.candidate_id,
    )
    assert foreign_candidate.proposal.idempotency_key == (
        f"{case.case_id}-candidate-foreign-target-current-{case.target_slot:02d}"
    )
    assert foreign_revision.proposal.idempotency_key == (
        f"{case.case_id}-revision-foreign-target-current-{case.target_slot:02d}"
    )
    assert foreign_candidate.proposal.evidence_ids == (capture.foreign_evidence_id,)


def test_current_revision_supersedes_old_revision_on_same_memory_id(
    tmp_path: Path,
) -> None:
    case, _path, references, store = _build_case_graph(tmp_path)
    scope = references.capture.local_scope
    old = store.get_revision(scope, references.revisions.target_old_revision_id)
    current = store.get_revision(
        scope,
        references.revisions.target_current_revision_id,
    )
    masked = store.get_revision(
        scope,
        references.revisions.target_masked_revision_id,
    )

    assert old.proposal.operation is RevisionOperation.ADD
    assert old.proposal.parent_revision_id is None
    assert old.generation == 0
    assert current.proposal.operation is RevisionOperation.SUPERSEDE
    assert current.proposal.parent_revision_id == old.revision_id
    assert current.generation == 1
    assert current.memory_id == old.memory_id
    assert masked.proposal.operation is RevisionOperation.ADD
    assert masked.generation == 0
    assert masked.memory_id != old.memory_id
    assert store.get_candidate(scope, old.proposal.candidate_id).proposal.content == (
        f"{case.target_key} = {case.old_value}"
    )
    assert store.get_candidate(
        scope, current.proposal.candidate_id
    ).proposal.content == (f"{case.target_key} = {case.current_value}")
    assert store.get_candidate(
        scope, masked.proposal.candidate_id
    ).proposal.content == (f"{case.target_key} = {case.masked_value}")


def test_release_alias_keys_scopes_and_call_order_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, MemoryScope, tuple[str, ...]]] = []
    append_release = SQLiteMemoryStore.append_release

    def append_release_spy(
        self: SQLiteMemoryStore,
        manifest,
        *,
        idempotency_key: str,
    ):
        calls.append((idempotency_key, manifest.scope, manifest.revision_ids))
        return append_release(self, manifest, idempotency_key=idempotency_key)

    monkeypatch.setattr(SQLiteMemoryStore, "append_release", append_release_spy)

    case, _path, references, _store = _build_case_graph(tmp_path)

    assert tuple(call[0] for call in calls) == (
        f"{case.case_id}-release-local-stale",
        f"{case.case_id}-release-local-current",
        f"{case.case_id}-release-local-masked",
        f"{case.case_id}-release-local-empty",
        f"{case.case_id}-release-foreign-sentinel",
    )
    assert tuple(call[1] for call in calls) == (
        references.capture.local_scope,
        references.capture.local_scope,
        references.capture.local_scope,
        references.capture.local_scope,
        references.capture.foreign_scope,
    )
    assert tuple(len(call[2]) for call in calls) == (6, 6, 6, 0, 1)


def test_release_membership_matches_current_stale_masked_and_empty_manifests(
    tmp_path: Path,
) -> None:
    case, _path, references, store = _build_case_graph(tmp_path)
    scope = references.capture.local_scope
    revisions = references.revisions
    releases = references.releases
    expected = {
        releases.stale_release_id: _expected_release_payloads(
            case,
            target_value=case.old_value,
        ),
        releases.current_release_id: _expected_release_payloads(
            case,
            target_value=case.current_value,
        ),
        releases.masked_release_id: _expected_release_payloads(
            case,
            target_value=case.masked_value,
        ),
        releases.empty_release_id: (),
    }

    for release_id, expected_payloads in expected.items():
        release = store.get_release(scope, release_id)
        members = store.get_release_revisions(scope, release_id)
        assert release.manifest.revision_ids == tuple(
            member.revision_id for member in members
        )
        actual_payloads: list[str] = []
        for member in members:
            candidate = store.get_candidate(scope, member.proposal.candidate_id)
            assert len(candidate.proposal.evidence_ids) == 1
            evidence = store.get(scope, candidate.proposal.evidence_ids[0])
            assert candidate.proposal.content == evidence.event.payload
            actual_payloads.append(candidate.proposal.content)
        assert tuple(actual_payloads) == expected_payloads

    assert len(expected[releases.current_release_id]) == 6
    assert (
        revisions.target_old_revision_id
        not in store.get_release(
            scope,
            releases.current_release_id,
        ).manifest.revision_ids
    )
    assert (
        revisions.target_current_revision_id
        not in store.get_release(
            scope,
            releases.stale_release_id,
        ).manifest.revision_ids
    )

    foreign_release = store.get_release(
        references.capture.foreign_scope,
        releases.foreign_sentinel_release_id,
    )
    foreign_members = store.get_release_revisions(
        references.capture.foreign_scope,
        releases.foreign_sentinel_release_id,
    )
    assert foreign_release.manifest.revision_ids == tuple(
        member.revision_id for member in foreign_members
    )
    assert len(foreign_members) == 1
    foreign_candidate = store.get_candidate(
        references.capture.foreign_scope,
        foreign_members[0].proposal.candidate_id,
    )
    assert foreign_candidate.proposal.content == (
        f"{case.target_key} = {case.current_value}"
    )
    assert foreign_candidate.proposal.evidence_ids == (
        references.capture.foreign_evidence_id,
    )
    assert (
        store.get(
            references.capture.foreign_scope,
            references.capture.foreign_evidence_id,
        ).event.payload
        == f"{case.target_key} = {case.current_value}"
    )


def test_rebuilding_same_database_is_an_exact_retry_without_new_rows(
    tmp_path: Path,
) -> None:
    case, path, first, store = _build_case_graph(tmp_path)
    before = (
        len(store.list(first.capture.local_scope)),
        len(store.list_candidates(first.capture.local_scope)),
        len(store.list_revisions(first.capture.local_scope)),
        len(store.list_releases(first.capture.local_scope)),
        len(store.list(first.capture.foreign_scope)),
        len(store.list_candidates(first.capture.foreign_scope)),
        len(store.list_revisions(first.capture.foreign_scope)),
        len(store.list_releases(first.capture.foreign_scope)),
    )

    second = helpfulness.build_case_database(case, path)

    assert second == first
    reopened = SQLiteMemoryStore(path)
    assert (
        len(reopened.list(first.capture.local_scope)),
        len(reopened.list_candidates(first.capture.local_scope)),
        len(reopened.list_revisions(first.capture.local_scope)),
        len(reopened.list_releases(first.capture.local_scope)),
        len(reopened.list(first.capture.foreign_scope)),
        len(reopened.list_candidates(first.capture.foreign_scope)),
        len(reopened.list_revisions(first.capture.foreign_scope)),
        len(reopened.list_releases(first.capture.foreign_scope)),
    ) == before


def test_reopening_sqlite_preserves_exact_graph(tmp_path: Path) -> None:
    _case, path, references, original = _build_case_graph(tmp_path)
    capture = references.capture
    local_revision_ids = _local_revision_ids(references)
    release_ids = (
        references.releases.stale_release_id,
        references.releases.current_release_id,
        references.releases.masked_release_id,
        references.releases.empty_release_id,
    )
    evidence_before = tuple(
        (record.evidence_id, record.content_hash, record.event.canonical_bytes())
        for record in original.list(capture.local_scope)
    )
    revisions_before = tuple(
        original.get_revision(capture.local_scope, revision_id)
        for revision_id in local_revision_ids
    )
    candidates_before = tuple(
        original.get_candidate(capture.local_scope, revision.proposal.candidate_id)
        for revision in revisions_before
    )
    releases_before = tuple(
        original.get_release(capture.local_scope, release_id)
        for release_id in release_ids
    )

    reopened = SQLiteMemoryStore(path)

    assert (
        tuple(
            (record.evidence_id, record.content_hash, record.event.canonical_bytes())
            for record in reopened.list(capture.local_scope)
        )
        == evidence_before
    )
    assert (
        tuple(
            reopened.get_revision(capture.local_scope, revision_id)
            for revision_id in local_revision_ids
        )
        == revisions_before
    )
    assert (
        tuple(
            reopened.get_candidate(capture.local_scope, revision.proposal.candidate_id)
            for revision in revisions_before
        )
        == candidates_before
    )
    assert (
        tuple(
            reopened.get_release(capture.local_scope, release_id)
            for release_id in release_ids
        )
        == releases_before
    )
    assert reopened.get_release(
        capture.foreign_scope,
        references.releases.foreign_sentinel_release_id,
    ) == original.get_release(
        capture.foreign_scope,
        references.releases.foreign_sentinel_release_id,
    )


def test_foreign_sentinel_release_is_not_found_in_local_scope(tmp_path: Path) -> None:
    _case, _path, references, store = _build_case_graph(tmp_path)
    foreign_release_id = references.releases.foreign_sentinel_release_id

    assert (
        store.get_release(
            references.capture.foreign_scope, foreign_release_id
        ).release_id
        == foreign_release_id
    )
    with pytest.raises(ReleaseNotFoundError) as error:
        store.get_release(references.capture.local_scope, foreign_release_id)
    assert error.type is ReleaseNotFoundError


def _release_source_setup(tmp_path: Path):
    case, path, references, store = _build_case_graph(tmp_path)
    assignment = helpfulness.ReleaseSourceAssignment(
        scope=references.capture.local_scope,
        release_id=references.releases.current_release_id,
    )
    audit = helpfulness.ReadAuditSink()
    capability = helpfulness.ReleaseReadCapability(
        store,
        assignment,
        audit,
    )
    return case, path, references, store, assignment, audit, capability


def _raw_source_setup(tmp_path: Path):
    case, path, references, store = _build_case_graph(tmp_path)
    assignment = helpfulness.RawSourceAssignment(
        scope=references.capture.local_scope,
        cutoff=references.capture.raw_history_cutoff,
    )
    audit = helpfulness.ReadAuditSink()
    capability = helpfulness.RawEvidenceReadCapability(
        store,
        assignment,
        audit,
    )
    return case, path, references, store, assignment, audit, capability


def _oracle_source_setup(tmp_path: Path):
    case, path, references, _store = _build_case_graph(tmp_path)
    entries = _case_entries(
        case,
        target_value=case.current_value,
        source_kind="oracle",
    )
    assignment = helpfulness.OracleSourceAssignment(
        scope=references.capture.local_scope,
        entries=entries,
    )
    audit = helpfulness.ReadAuditSink()
    capability = helpfulness.OracleEntryCapability(assignment, audit)
    return case, path, references, assignment, audit, capability


def test_release_capability_exposes_only_assigned_reachable_graph(
    tmp_path: Path,
) -> None:
    (
        case,
        path,
        references,
        store,
        assignment,
        audit,
        capability,
    ) = _release_source_setup(tmp_path)

    assert not hasattr(capability, "list_releases")
    assert not hasattr(capability, "list_candidates")
    assert not hasattr(capability, "list_revisions")
    assert not hasattr(capability, "database_path")
    treatment = helpfulness.resolve_treatment(capability)
    helpfulness.validate_resolved_treatment(
        path,
        assignment,
        treatment,
        audit.snapshot(),
    )
    assigned = store.get_release(assignment.scope, assignment.release_id)
    assert treatment.release_id == assignment.release_id
    assert treatment.eligible_ids == assigned.manifest.revision_ids
    assert treatment.retrieved_ids == assigned.manifest.revision_ids
    assert treatment.returned_ids == assigned.manifest.revision_ids
    assert len(treatment.entries) == 6
    assert tuple(entry.slot for entry in treatment.entries) == tuple(range(6))
    assert tuple(f"{entry.key} = {entry.value}" for entry in treatment.entries) == (
        _expected_release_payloads(case, target_value=case.current_value)
    )
    assert not hasattr(treatment, "injected_ids")

    denied_audit = helpfulness.ReadAuditSink()
    denied = helpfulness.ReleaseReadCapability(store, assignment, denied_audit)
    denied.get_assigned_release()
    with pytest.raises(helpfulness.UnauthorizedReadError):
        denied.get_revision(references.revisions.target_old_revision_id)
    denied_event = denied_audit.snapshot()[-1]
    assert denied_event.operation == "get_revision"
    assert denied_event.allowed is False
    assert denied_event.requested_ids == (references.revisions.target_old_revision_id,)
    assert denied_event.returned_record_ids == ()
    assert denied_event.returned_content_hashes == ()

    before_interface_attempt = denied_audit.snapshot()
    with pytest.raises(AttributeError):
        denied.list_releases()  # type: ignore[attr-defined]
    assert denied_audit.snapshot() == before_interface_attempt


def test_raw_capability_exposes_only_fixed_cutoff_policy(tmp_path: Path) -> None:
    (
        _case,
        path,
        references,
        _store,
        assignment,
        audit,
        capability,
    ) = _raw_source_setup(tmp_path)

    assert not hasattr(capability, "get_revision")
    assert not hasattr(capability, "get_candidate")
    assert not hasattr(capability, "get_assigned_release")
    assert not hasattr(capability, "database_path")
    treatment = helpfulness.resolve_treatment(capability)
    helpfulness.validate_resolved_treatment(
        path,
        assignment,
        treatment,
        audit.snapshot(),
    )
    assert treatment.source_kind == "raw_evidence"
    assert treatment.release_id is None
    assert treatment.eligible_ids == treatment.source_evidence_ids
    assert treatment.retrieved_ids == treatment.source_evidence_ids
    assert treatment.returned_ids == treatment.source_evidence_ids
    assert all(entry.revision_id is None for entry in treatment.entries)
    assert all(entry.candidate_id is None for entry in treatment.entries)


def test_raw_cutoff_returns_only_six_user_feedback_records_in_chronology(
    tmp_path: Path,
) -> None:
    (
        case,
        _path,
        references,
        store,
        assignment,
        audit,
        capability,
    ) = _raw_source_setup(tmp_path)
    native_records = store.list(references.capture.local_scope)

    treatment = helpfulness.resolve_treatment(capability)

    chronological = tuple(
        sorted(
            (
                record
                for record in native_records
                if record.event.kind
                in {EvidenceKind.USER_MESSAGE, EvidenceKind.FEEDBACK}
                and record.event.observed_at <= references.capture.raw_history_cutoff
            ),
            key=lambda record: (
                record.event.observed_at,
                record.event.sequence_no,
                record.evidence_id,
            ),
        )
    )
    assert tuple(record.evidence_id for record in native_records[:6]) != tuple(
        record.evidence_id for record in chronological
    )
    assert treatment.source_evidence_ids == tuple(
        record.evidence_id for record in chronological
    )
    assert len(treatment.entries) == 6
    assert tuple(entry.value for entry in treatment.entries) == (
        *(
            entry.value
            for entry in _core_entries(
                case,
                target_value=case.old_value,
                source_kind="raw_evidence",
            )
        ),
        case.current_value,
    )
    target_entries = tuple(
        entry for entry in treatment.entries if entry.key == case.target_key
    )
    assert tuple(entry.value for entry in target_entries) == (
        case.old_value,
        case.current_value,
    )
    assert target_entries[0].slot == target_entries[1].slot == case.target_slot
    assert tuple(event.operation for event in audit.snapshot()) == (
        "list_eligible_evidence",
    )
    raw_event = audit.snapshot()[0]
    assert raw_event.requested_scope == assignment.scope
    assert raw_event.requested_ids == ()
    assert raw_event.allowed is True
    assert raw_event.returned_record_ids == tuple(
        record.evidence_id for record in chronological
    )
    assert raw_event.returned_content_hashes == tuple(
        record.content_hash for record in chronological
    )


def test_raw_policy_rejects_before_cutoff_kind_and_after_cutoff_kind_sentinels(
    tmp_path: Path,
) -> None:
    (
        case,
        path,
        references,
        store,
        assignment,
        audit,
        capability,
    ) = _raw_source_setup(tmp_path)
    before_cutoff_environment = store.append(
        EvidenceEvent(
            scope=assignment.scope,
            session_id=f"{case.case_id}-sentinel-kind",
            run_id=f"{case.case_id}-run-sentinel-kind",
            sequence_no=0,
            kind=EvidenceKind.ENVIRONMENT,
            payload=f"{case.padding_entry.key} = {case.padding_entry.value}",
            observed_at=assignment.cutoff - timedelta(seconds=1),
            idempotency_key=f"{case.case_id}-evidence-sentinel-kind",
        )
    )
    after_cutoff_user = store.append(
        EvidenceEvent(
            scope=assignment.scope,
            session_id=f"{case.case_id}-sentinel-cutoff-user",
            run_id=f"{case.case_id}-run-sentinel-cutoff-user",
            sequence_no=0,
            kind=EvidenceKind.USER_MESSAGE,
            payload=f"{case.target_key} = {case.current_value}",
            observed_at=assignment.cutoff + timedelta(seconds=1),
            idempotency_key=f"{case.case_id}-evidence-sentinel-cutoff-user",
        )
    )
    after_cutoff_feedback = store.append(
        EvidenceEvent(
            scope=assignment.scope,
            session_id=f"{case.case_id}-sentinel-cutoff-feedback",
            run_id=f"{case.case_id}-run-sentinel-cutoff-feedback",
            sequence_no=0,
            kind=EvidenceKind.FEEDBACK,
            payload=f"{case.shared_entries[0].key} = {case.shared_entries[0].value}",
            observed_at=assignment.cutoff + timedelta(seconds=2),
            idempotency_key=f"{case.case_id}-evidence-sentinel-cutoff-feedback",
        )
    )

    treatment = helpfulness.resolve_treatment(capability)
    helpfulness.validate_resolved_treatment(
        path,
        assignment,
        treatment,
        audit.snapshot(),
    )

    assert treatment.source_evidence_ids == (
        *references.capture.old_evidence_ids,
        references.capture.current_evidence_id,
    )
    assert {
        before_cutoff_environment.evidence_id,
        after_cutoff_user.evidence_id,
        after_cutoff_feedback.evidence_id,
    }.isdisjoint(treatment.source_evidence_ids)


def test_oracle_capability_has_no_store_access(tmp_path: Path) -> None:
    case, path, _references, assignment, audit, capability = _oracle_source_setup(
        tmp_path
    )

    assert not hasattr(capability, "store")
    assert not hasattr(capability, "_store")
    assert not hasattr(capability, "get_revision")
    assert not hasattr(capability, "get_candidate")
    treatment = helpfulness.resolve_treatment(capability)
    helpfulness.validate_resolved_treatment(
        path,
        assignment,
        treatment,
        audit.snapshot(),
    )
    assert treatment.entries == assignment.entries
    assert treatment.release_id is None
    assert treatment.eligible_ids == ()
    assert treatment.retrieved_ids == ()
    assert treatment.returned_ids == ()
    assert treatment.source_evidence_ids == ()
    assert tuple(entry.value for entry in treatment.entries) == tuple(
        entry.value
        for entry in _case_entries(
            case,
            target_value=case.current_value,
            source_kind="oracle",
        )
    )
    oracle_event = audit.snapshot()[0]
    assert oracle_event.operation == "entries"
    assert oracle_event.requested_scope == assignment.scope
    assert oracle_event.requested_ids == ()
    assert oracle_event.allowed is True
    assert oracle_event.returned_record_ids == ()
    assert oracle_event.returned_content_hashes == tuple(
        sha256(f"{entry.key}\t{entry.value}".encode()).hexdigest()
        for entry in assignment.entries
    )


def test_audit_sequence_scope_ids_and_hashes_are_exact(tmp_path: Path) -> None:
    (
        _case,
        path,
        references,
        store,
        assignment,
        audit,
        capability,
    ) = _release_source_setup(tmp_path)

    treatment = helpfulness.resolve_treatment(capability)
    events = audit.snapshot()
    release = store.get_release(assignment.scope, assignment.release_id)
    expected_operations = ["get_assigned_release"]
    expected_ids = [(release.release_id,)]
    expected_hashes = [(release.content_hash,)]
    for revision_id in release.manifest.revision_ids:
        revision = store.get_revision(assignment.scope, revision_id)
        candidate = store.get_candidate(
            assignment.scope,
            revision.proposal.candidate_id,
        )
        expected_operations.extend(("get_revision", "get_candidate"))
        expected_ids.extend(((revision.revision_id,), (candidate.candidate_id,)))
        expected_hashes.extend(((revision.content_hash,), (candidate.content_hash,)))

    assert tuple(event.operation for event in events) == tuple(expected_operations)
    assert all(event.requested_scope == assignment.scope for event in events)
    assert all(event.allowed is True for event in events)
    assert tuple(event.requested_ids for event in events) == tuple(expected_ids)
    assert tuple(event.returned_record_ids for event in events) == tuple(expected_ids)
    assert tuple(event.returned_content_hashes for event in events) == tuple(
        expected_hashes
    )
    assert treatment.source_evidence_ids == tuple(
        evidence_id for entry in treatment.entries for evidence_id in entry.evidence_ids
    )

    empty_assignment = helpfulness.ReleaseSourceAssignment(
        scope=assignment.scope,
        release_id=references.releases.empty_release_id,
    )
    empty_audit = helpfulness.ReadAuditSink()
    empty_capability = helpfulness.ReleaseReadCapability(
        store,
        empty_assignment,
        empty_audit,
    )
    empty_treatment = helpfulness.resolve_treatment(empty_capability)
    helpfulness.validate_resolved_treatment(
        path,
        empty_assignment,
        empty_treatment,
        empty_audit.snapshot(),
    )
    assert empty_treatment.entries == ()
    assert tuple(event.operation for event in empty_audit.snapshot()) == (
        "get_assigned_release",
    )


@pytest.mark.parametrize("mutation", ("swap", "hash"))
def test_same_length_wrong_audit_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    (
        _case,
        path,
        _references,
        _store,
        assignment,
        audit,
        capability,
    ) = _release_source_setup(tmp_path)
    treatment = helpfulness.resolve_treatment(capability)
    valid = audit.snapshot()
    if mutation == "swap":
        wrong = (valid[0], valid[2], valid[1], *valid[3:])
    else:
        wrong = (
            valid[0],
            replace(valid[1], returned_content_hashes=("0" * 64,)),
            *valid[2:],
        )

    assert len(wrong) == len(valid)
    with pytest.raises(helpfulness.TreatmentValidationError) as error:
        helpfulness.validate_resolved_treatment(
            path,
            assignment,
            treatment,
            wrong,
        )
    assert error.value.reason == "source_or_audit_mismatch"


def test_entries_must_be_independently_reachable_from_assigned_source(
    tmp_path: Path,
) -> None:
    (
        _case,
        path,
        _references,
        _store,
        assignment,
        audit,
        capability,
    ) = _release_source_setup(tmp_path)
    treatment = helpfulness.resolve_treatment(capability)
    forged = replace(
        treatment,
        entries=(
            replace(treatment.entries[0], candidate_id="cand_forged"),
            *treatment.entries[1:],
        ),
    )

    with pytest.raises(helpfulness.TreatmentValidationError) as provenance_error:
        helpfulness.validate_resolved_treatment(
            path,
            assignment,
            forged,
            audit.snapshot(),
        )
    assert provenance_error.value.reason == "provenance_mismatch"

    extra_audit = helpfulness.ReadAuditSink()
    extra_capability = helpfulness.ReleaseReadCapability(
        SQLiteMemoryStore(path),
        assignment,
        extra_audit,
    )
    extra_capability.get_assigned_release()
    otherwise_valid = helpfulness.resolve_treatment(extra_capability)
    with pytest.raises(helpfulness.TreatmentValidationError) as extra_error:
        helpfulness.validate_resolved_treatment(
            path,
            assignment,
            otherwise_valid,
            extra_audit.snapshot(),
        )
    assert extra_error.value.reason == "extra_read"


class LatestReleaseResolver:
    def __call__(self, capability):
        return capability.list_releases()


class AllCandidatesResolver:
    def __init__(self, candidate_id: str) -> None:
        self._candidate_id = candidate_id

    def __call__(self, capability):
        capability.get_assigned_release()
        return capability.get_candidate(self._candidate_id)


class PeekThenDiscardResolver:
    def __call__(self, capability):
        capability.get_assigned_release()
        return helpfulness.resolve_treatment(capability)


class ForgedProvenanceResolver:
    def __call__(self, capability):
        treatment = helpfulness.resolve_treatment(capability)
        return replace(
            treatment,
            entries=(
                replace(treatment.entries[0], revision_id="rev_forged"),
                *treatment.entries[1:],
            ),
        )


class RawViaRevisionResolver:
    def __call__(self, capability):
        return capability.get_revision("rev_forbidden")


class SourceSwapResolver:
    def __call__(self, capability):
        return helpfulness.resolve_treatment(capability)


class InternalAttributeBugResolver:
    def __call__(self, capability):
        del capability
        return self.missing_internal_attribute


class ProxyPrivateStateBugResolver:
    def __call__(self, capability):
        return capability._missing_proxy_private_state


def test_capability_interface_error_does_not_swallow_resolver_attribute_bugs(
    tmp_path: Path,
) -> None:
    (
        _case,
        path,
        _references,
        _store,
        assignment,
        audit,
        capability,
    ) = _release_source_setup(tmp_path)

    before = audit.snapshot()
    with pytest.raises(AttributeError) as direct_interface_error:
        capability.list_releases()  # type: ignore[attr-defined]
    assert type(direct_interface_error.value) is AttributeError
    assert audit.snapshot() == before

    with pytest.raises(AttributeError) as error:
        helpfulness.resolve_and_validate(
            InternalAttributeBugResolver(),
            capability,
            database_path=path,
            assignment=assignment,
            audit_sink=audit,
        )
    assert type(error.value) is AttributeError

    with pytest.raises(AttributeError) as proxy_error:
        helpfulness.resolve_and_validate(
            ProxyPrivateStateBugResolver(),
            capability,
            database_path=path,
            assignment=assignment,
            audit_sink=audit,
        )
    assert type(proxy_error.value) is AttributeError

    unbound_proxy = object.__new__(helpfulness._CapabilityBoundary)
    with pytest.raises(AttributeError) as unbound_error:
        helpfulness.resolve_treatment(unbound_proxy)
    assert type(unbound_error.value) is AttributeError


def test_capability_missing_private_state_is_not_misclassified_as_interface(
    tmp_path: Path,
) -> None:
    (
        _case,
        path,
        _references,
        store,
        assignment,
        audit,
        capability,
    ) = _release_source_setup(tmp_path)
    object.__delattr__(capability, "_ReleaseReadCapability__store")

    with pytest.raises(AttributeError) as direct_error:
        helpfulness.resolve_treatment(capability)
    assert type(direct_error.value) is AttributeError

    runner_audit = helpfulness.ReadAuditSink()
    runner_capability = helpfulness.ReleaseReadCapability(
        store,
        assignment,
        runner_audit,
    )
    object.__delattr__(runner_capability, "_ReleaseReadCapability__store")
    with pytest.raises(AttributeError) as runner_error:
        helpfulness.resolve_and_validate(
            helpfulness.resolve_treatment,
            runner_capability,
            database_path=path,
            assignment=assignment,
            audit_sink=runner_audit,
        )
    assert type(runner_error.value) is AttributeError


@pytest.mark.parametrize(
    ("mutant", "reason"),
    (
        ("latest", "capability_interface_violation"),
        ("all_candidates", "unauthorized_read"),
        ("peek", "extra_read"),
        ("forged", "provenance_mismatch"),
        ("raw_via_revision", "capability_interface_violation"),
        ("source_swap", "source_or_audit_mismatch"),
    ),
)
def test_source_mutants_fail_for_pre_registered_reason(
    tmp_path: Path,
    mutant: str,
    reason: str,
) -> None:
    (
        _case,
        path,
        references,
        store,
        release_assignment,
        release_audit,
        release_capability,
    ) = _release_source_setup(tmp_path)
    assignment = release_assignment
    audit = release_audit
    capability = release_capability
    if mutant == "latest":
        resolver = LatestReleaseResolver()
    elif mutant == "all_candidates":
        stale = store.get_revision(
            references.capture.local_scope,
            references.revisions.target_old_revision_id,
        )
        resolver = AllCandidatesResolver(stale.proposal.candidate_id)
    elif mutant == "peek":
        resolver = PeekThenDiscardResolver()
    elif mutant == "forged":
        resolver = ForgedProvenanceResolver()
    elif mutant == "raw_via_revision":
        (tmp_path / "raw").mkdir()
        _case, path, _references, _store, assignment, audit, capability = (
            _raw_source_setup(tmp_path / "raw")
        )
        resolver = RawViaRevisionResolver()
    else:
        (tmp_path / "swap").mkdir()
        _case, path, _references, _store, raw_assignment, audit, capability = (
            _raw_source_setup(tmp_path / "swap")
        )
        assignment = release_assignment
        assert raw_assignment.scope == assignment.scope
        resolver = SourceSwapResolver()

    with pytest.raises(helpfulness.TreatmentValidationError) as error:
        helpfulness.resolve_and_validate(
            resolver,
            capability,
            database_path=path,
            assignment=assignment,
            audit_sink=audit,
        )
    assert error.value.reason == reason


def _task5_bundle(tmp_path: Path, *, arm: str, execution_index: int = 0):
    case, path, references, store = _build_case_graph(tmp_path)
    audit = helpfulness.ReadAuditSink()
    if arm == "raw_history":
        assignment = helpfulness.RawSourceAssignment(
            scope=references.capture.local_scope,
            cutoff=references.capture.raw_history_cutoff,
        )
        capability = helpfulness.RawEvidenceReadCapability(store, assignment, audit)
    elif arm == "oracle":
        assignment = helpfulness.OracleSourceAssignment(
            scope=references.capture.local_scope,
            entries=_case_entries(
                case,
                target_value=case.current_value,
                source_kind="oracle",
            ),
        )
        capability = helpfulness.OracleEntryCapability(assignment, audit)
    else:
        release_id = {
            "current_release": references.releases.current_release_id,
            "memory_off": references.releases.empty_release_id,
            "target_masked": references.releases.masked_release_id,
            "stale_release": references.releases.stale_release_id,
        }[arm]
        assignment = helpfulness.ReleaseSourceAssignment(
            scope=references.capture.local_scope,
            release_id=release_id,
        )
        capability = helpfulness.ReleaseReadCapability(store, assignment, audit)
    treatment = helpfulness.resolve_treatment(capability)
    helpfulness.validate_resolved_treatment(
        path,
        assignment,
        treatment,
        audit.snapshot(),
    )
    rendered = helpfulness.render_context(treatment.entries)
    query = _query(case)
    consumer_result = helpfulness.consume_scripted(query, rendered.bytes)
    observation = helpfulness.make_execution_observation(
        execution_index=execution_index,
        treatment=treatment,
        reader_audit=audit.snapshot(),
        rendered_context=rendered,
        query=query,
        consumer_result=consumer_result,
        model_call_receipt=None,
        capture_session_ids=references.capture.capture_session_ids,
        future_session_id=f"{case.case_id}-future-session-{execution_index:03d}",
        future_run_id=f"{case.case_id}-future-run-{execution_index:03d}",
        capture_pid=101,
        future_pid=202,
        capture_process_instance_id="capture-instance",
        future_process_instance_id="future-instance",
    )
    schedule = helpfulness.make_parent_schedule_item(
        execution_index=execution_index,
        case=case,
        references=references,
        arm=arm,
    )
    return {
        "case": case,
        "path": path,
        "references": references,
        "assignment": assignment,
        "audit": audit.snapshot(),
        "treatment": treatment,
        "rendered": rendered,
        "query": query,
        "consumer_result": consumer_result,
        "observation": observation,
        "schedule": schedule,
    }


def _observation_with_rendered(bundle, rendered, consumer_result, *, model=None):
    original = bundle["observation"]
    return helpfulness.make_execution_observation(
        execution_index=original.execution_index,
        treatment=bundle["treatment"],
        reader_audit=bundle["audit"],
        rendered_context=rendered,
        query=bundle["query"],
        consumer_result=consumer_result,
        model_call_receipt=model,
        capture_session_ids=original.capture_session_ids,
        future_session_id=original.future_session_id,
        future_run_id=original.future_run_id,
        capture_pid=original.capture_pid,
        future_pid=original.future_pid,
        capture_process_instance_id=original.capture_process_instance_id,
        future_process_instance_id=original.future_process_instance_id,
    )


def test_ids_are_not_injected_before_consumer_receipt_matches(tmp_path: Path) -> None:
    bundle = _task5_bundle(tmp_path, arm="current_release")
    observation = bundle["observation"]

    observation_fields = {field.name for field in fields(observation)}
    assert "injected_ids" not in observation_fields
    assert "injected_revision_ids" not in observation_fields

    (trace,) = helpfulness.parent_join_and_score(
        (observation,),
        (bundle["schedule"],),
        enforce_scripted_outcomes=True,
    )
    assert trace.injected_revision_ids == observation.returned_ids


@pytest.mark.parametrize(
    ("receipt_field", "reason"),
    (
        ("received_context_sha256", "received_context_mismatch"),
        ("received_context_utf8_bytes", "received_context_mismatch"),
        ("received_query_sha256", "received_query_mismatch"),
    ),
)
def test_context_or_query_receipt_mismatch_invalidates_before_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_field: str,
    reason: str,
) -> None:
    bundle = _task5_bundle(tmp_path, arm="current_release")
    observation = bundle["observation"]
    receipt = observation.consumer_input_receipt
    wrong_value = (
        receipt.received_context_utf8_bytes + 1
        if receipt_field == "received_context_utf8_bytes"
        else "0" * 64
    )
    wrong_observation = replace(
        observation,
        consumer_input_receipt=replace(receipt, **{receipt_field: wrong_value}),
    )

    def scoring_must_not_run(_response: str) -> str:
        raise AssertionError("normalization ran before receipt validation")

    monkeypatch.setattr(helpfulness, "normalize_response", scoring_must_not_run)
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (wrong_observation,),
            (bundle["schedule"],),
            enforce_scripted_outcomes=False,
        )
    assert error.value.reason == reason


def test_model_call_receipt_binds_exact_memory_slice_and_token_ids(
    tmp_path: Path,
) -> None:
    bundle = _task5_bundle(tmp_path, arm="current_release")
    rendered = bundle["rendered"]
    prefix = b"system\n"
    suffix = b"\nuser"
    prompt = prefix + rendered.bytes + suffix
    token_ids = (101, 202, 303)
    receipt = helpfulness.make_model_call_receipt(
        submitted_prompt=prompt,
        context_start=len(prefix),
        context_end=len(prefix) + len(rendered.bytes),
        input_token_ids=token_ids,
    )

    assert receipt.submitted_prompt_sha256 == sha256(prompt).hexdigest()
    assert receipt.submitted_prompt_context_sha256 == sha256(rendered.bytes).hexdigest()
    assert (
        receipt.submitted_input_token_ids_sha256 == sha256(b"[101,202,303]").hexdigest()
    )
    assert receipt.submitted_input_token_count == 3
    observation = replace(bundle["observation"], model_call_receipt=receipt)
    (trace,) = helpfulness.parent_join_and_score(
        (observation,),
        (bundle["schedule"],),
        enforce_scripted_outcomes=True,
    )
    assert trace.submitted_prompt_sha256 == receipt.submitted_prompt_sha256
    assert trace.submitted_prompt_context_start == len(prefix)
    assert trace.submitted_prompt_context_end == len(prefix) + len(rendered.bytes)
    assert (
        trace.submitted_input_token_ids_sha256
        == receipt.submitted_input_token_ids_sha256
    )

    invalid_receipt = replace(
        receipt,
        submitted_input_token_ids_sha256="z" * 64,
    )
    invalid_observation = replace(
        bundle["observation"],
        model_call_receipt=invalid_receipt,
    )
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (invalid_observation,),
            (bundle["schedule"],),
            enforce_scripted_outcomes=False,
        )
    assert error.value.reason == "call_boundary_token_mismatch"


def test_child_observation_has_no_arm_expected_response_or_utility(
    tmp_path: Path,
) -> None:
    observation = _task5_bundle(tmp_path, arm="current_release")["observation"]
    names = {field.name for field in fields(observation)}

    assert names.isdisjoint(
        {
            "arm",
            "case_id",
            "expected_response",
            "normalized_response",
            "utility",
            "abstained",
            "followed_injected_value",
            "injected_revision_ids",
        }
    )


def test_parent_rejects_missing_duplicate_or_swapped_execution_indexes(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    current = _task5_bundle(first_dir, arm="current_release", execution_index=0)
    stale = _task5_bundle(second_dir, arm="stale_release", execution_index=1)
    observations = (current["observation"], stale["observation"])
    schedule = (current["schedule"], stale["schedule"])

    assert (
        len(
            helpfulness.parent_join_and_score(
                observations,
                schedule,
                enforce_scripted_outcomes=True,
            )
        )
        == 2
    )
    invalid = (
        ((observations[0],), "execution_index_mismatch"),
        ((observations[0], observations[0]), "execution_index_mismatch"),
        (
            (*observations, replace(observations[0], execution_index=99)),
            "execution_index_mismatch",
        ),
        (
            (
                replace(observations[0], execution_index=1),
                replace(observations[1], execution_index=0),
            ),
            "assignment_mismatch",
        ),
        (
            (replace(observations[0], source_kind="raw_evidence"), observations[1]),
            "assignment_mismatch",
        ),
        (
            (
                replace(
                    observations[0],
                    release_id=observations[1].release_id,
                ),
                observations[1],
            ),
            "assignment_mismatch",
        ),
    )
    for candidate_observations, reason in invalid:
        with pytest.raises(helpfulness.ObservationValidationError) as error:
            helpfulness.parent_join_and_score(
                candidate_observations,
                schedule,
                enforce_scripted_outcomes=False,
            )
        assert error.value.reason == reason


def test_parent_validates_entire_batch_before_any_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _task5_bundle(first_dir, arm="current_release", execution_index=0)
    second = _task5_bundle(second_dir, arm="stale_release", execution_index=1)
    second_observation = second["observation"]
    invalid_second = replace(
        second_observation,
        consumer_input_receipt=replace(
            second_observation.consumer_input_receipt,
            received_context_sha256="0" * 64,
        ),
    )

    def normalization_must_not_run(_response: str) -> str:
        raise AssertionError("normalization ran before whole-batch validity")

    monkeypatch.setattr(helpfulness, "normalize_response", normalization_must_not_run)
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (first["observation"], invalid_second),
            (first["schedule"], second["schedule"]),
            enforce_scripted_outcomes=False,
        )
    assert error.value.reason == "received_context_mismatch"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("audit_order", "source_or_audit_mismatch"),
        ("audit_hash", "source_or_audit_mismatch"),
        ("returned_ids", "provenance_mismatch"),
        ("source_evidence", "provenance_mismatch"),
        ("entry_same_context", "injection_provenance_mismatch"),
    ),
)
def test_parent_independently_rejects_same_length_forged_source_contracts(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    bundle = _task5_bundle(tmp_path, arm="current_release")
    observation = bundle["observation"]
    if mutation == "audit_order":
        audit = observation.reader_audit
        observation = replace(
            observation,
            reader_audit=(audit[0], audit[2], audit[1], *audit[3:]),
        )
    elif mutation == "audit_hash":
        audit = observation.reader_audit
        observation = replace(
            observation,
            reader_audit=(
                audit[0],
                replace(audit[1], returned_content_hashes=("0" * 64,)),
                *audit[2:],
            ),
        )
    elif mutation == "returned_ids":
        observation = replace(
            observation,
            returned_ids=(
                observation.returned_ids[1],
                observation.returned_ids[0],
                *observation.returned_ids[2:],
            ),
        )
    elif mutation == "source_evidence":
        observation = replace(
            observation,
            source_evidence_ids=(
                observation.source_evidence_ids[1],
                observation.source_evidence_ids[0],
                *observation.source_evidence_ids[2:],
            ),
        )
    else:
        first = observation.entries[0]
        forged_value = bundle["case"].old_value
        forged = replace(
            first,
            value=forged_value,
            content_sha256=sha256(f"{first.key}\t{forged_value}".encode()).hexdigest(),
        )
        observation = replace(
            observation,
            entries=(forged, *observation.entries[1:]),
        )

    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (observation,),
            (bundle["schedule"],),
            enforce_scripted_outcomes=False,
        )
    assert error.value.reason == reason


def test_parent_adds_normalization_expected_response_and_utility_after_join(
    tmp_path: Path,
) -> None:
    bundle = _task5_bundle(tmp_path, arm="current_release")
    observation = replace(
        bundle["observation"],
        response=f"  {bundle['case'].current_value.lower()}  ",
    )

    (trace,) = helpfulness.parent_join_and_score(
        (observation,),
        (bundle["schedule"],),
        enforce_scripted_outcomes=True,
    )

    assert trace.arm == "current_release"
    assert trace.expected_response == bundle["case"].current_value
    assert trace.normalized_response == bundle["case"].current_value
    assert trace.utility == 1
    assert trace.abstained is False


def test_followed_injected_value_requires_acknowledged_target_entry(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "current"
    masked_dir = tmp_path / "masked"
    current_dir.mkdir()
    masked_dir.mkdir()
    current = _task5_bundle(current_dir, arm="current_release")
    masked = _task5_bundle(masked_dir, arm="target_masked")

    (current_trace,) = helpfulness.parent_join_and_score(
        (current["observation"],),
        (current["schedule"],),
        enforce_scripted_outcomes=True,
    )
    (masked_trace,) = helpfulness.parent_join_and_score(
        (masked["observation"],),
        (masked["schedule"],),
        enforce_scripted_outcomes=True,
    )
    shared_value = next(
        entry.value
        for entry in current["treatment"].entries
        if entry.key != current["case"].target_key
    )
    shared_observation = replace(current["observation"], response=shared_value)
    (shared_trace,) = helpfulness.parent_join_and_score(
        (shared_observation,),
        (current["schedule"],),
        enforce_scripted_outcomes=False,
    )

    assert current_trace.followed_injected_value is True
    assert masked_trace.followed_injected_value is False
    assert shared_trace.followed_injected_value is False


@pytest.mark.parametrize(
    ("mutant", "reason"),
    (
        ("retrieved_equals_injected", "injection_provenance_mismatch"),
        ("drop_masked", "received_context_mismatch"),
        ("swap_context", "call_boundary_context_mismatch"),
        ("ignore_memory", "strict_outcome_failure"),
        ("first_occurrence_raw", "raw_order_failure"),
    ),
)
def test_exposure_mutants_fail_for_pre_registered_reason(
    tmp_path: Path,
    mutant: str,
    reason: str,
) -> None:
    arm = (
        "target_masked"
        if mutant == "drop_masked"
        else "raw_history"
        if mutant == "first_occurrence_raw"
        else "current_release"
    )
    bundle = _task5_bundle(tmp_path, arm=arm)
    observation = bundle["observation"]
    if mutant == "retrieved_equals_injected":
        rendered = helpfulness.render_context(bundle["treatment"].entries[:-1])
        result = helpfulness.consume_scripted(bundle["query"], rendered.bytes)
        observation = _observation_with_rendered(bundle, rendered, result)
    elif mutant == "drop_masked":
        rendered = helpfulness.render_context(
            tuple(
                entry
                for entry in bundle["treatment"].entries
                if entry.key != bundle["case"].target_key
            )
        )
        result = helpfulness.consume_scripted(bundle["query"], rendered.bytes)
        observation = replace(
            observation,
            consumer_input_receipt=result.input_receipt,
            response=result.response,
        )
    elif mutant == "swap_context":
        rendered = bundle["rendered"]
        swapped = b"X" * len(rendered.bytes)
        prefix = b"system\n"
        receipt = helpfulness.make_model_call_receipt(
            submitted_prompt=prefix + swapped,
            context_start=len(prefix),
            context_end=len(prefix) + len(swapped),
            input_token_ids=(7, 8, 9),
        )
        observation = replace(observation, model_call_receipt=receipt)
    elif mutant == "ignore_memory":
        observation = replace(observation, response="UNKNOWN")
    else:
        observation = replace(observation, response=bundle["case"].old_value)

    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (observation,),
            (bundle["schedule"],),
            enforce_scripted_outcomes=True,
        )
    assert error.value.reason == reason


@pytest.mark.parametrize(
    (
        "arm",
        "expected_source_kind",
        "expected_utility",
        "expected_followed",
    ),
    (
        ("current_release", "release", 1, True),
        ("raw_history", "raw_evidence", 1, True),
        ("memory_off", "release", 0, False),
        ("target_masked", "release", 0, False),
        ("stale_release", "release", -1, True),
        ("oracle", "oracle", 1, True),
    ),
)
def test_case_derived_six_arm_happy_paths_are_exact(
    tmp_path: Path,
    arm: str,
    expected_source_kind: str,
    expected_utility: int,
    expected_followed: bool,
) -> None:
    bundle = _task5_bundle(tmp_path, arm=arm)
    case = bundle["case"]
    references = bundle["references"]
    target_revision_id = {
        "current_release": references.revisions.target_current_revision_id,
        "target_masked": references.revisions.target_masked_revision_id,
        "stale_release": references.revisions.target_old_revision_id,
    }.get(arm)
    if target_revision_id is None:
        expected_injected_revision_ids = ()
    else:
        shared = iter(references.revisions.shared_revision_ids)
        expected_injected_revision_ids = tuple(
            target_revision_id if slot == case.target_slot else next(shared)
            for slot in range(5)
        ) + (references.revisions.padding_revision_id,)
    expected_release_id = {
        "current_release": references.releases.current_release_id,
        "raw_history": None,
        "memory_off": references.releases.empty_release_id,
        "target_masked": references.releases.masked_release_id,
        "stale_release": references.releases.stale_release_id,
        "oracle": None,
    }[arm]
    expected_response = {
        "current_release": case.current_value,
        "raw_history": case.current_value,
        "memory_off": "UNKNOWN",
        "target_masked": "UNKNOWN",
        "stale_release": case.old_value,
        "oracle": case.current_value,
    }[arm]

    (trace,) = helpfulness.parent_join_and_score(
        (bundle["observation"],),
        (bundle["schedule"],),
        enforce_scripted_outcomes=True,
    )

    assert trace.source_kind == expected_source_kind
    assert trace.release_id == expected_release_id
    assert trace.expected_response == expected_response
    assert trace.normalized_response == expected_response
    assert trace.utility == expected_utility
    assert trace.injected_revision_ids == expected_injected_revision_ids
    assert trace.followed_injected_value is expected_followed
    assert (
        trace.submitted_prompt_sha256,
        trace.submitted_prompt_context_start,
        trace.submitted_prompt_context_end,
        trace.submitted_prompt_context_sha256,
        trace.submitted_input_token_ids_sha256,
        trace.submitted_input_token_count,
    ) == (None, None, None, None, None, None)


def test_later_strict_failure_prevents_all_trace_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _task5_bundle(first_dir, arm="current_release", execution_index=0)
    second = _task5_bundle(second_dir, arm="stale_release", execution_index=1)
    invalid_second = replace(second["observation"], response="UNKNOWN")

    def trace_must_not_be_constructed(*_args, **_kwargs):
        raise AssertionError("trace construction ran before all strict outcomes passed")

    monkeypatch.setattr(
        helpfulness,
        "_trace_from_observation",
        trace_must_not_be_constructed,
    )
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (first["observation"], invalid_second),
            (first["schedule"], second["schedule"]),
            enforce_scripted_outcomes=True,
        )
    assert error.value.reason == "strict_outcome_failure"


def _wire_future_setup(tmp_path: Path):
    case, database_path, references, _store = _build_case_graph(tmp_path)
    source = helpfulness.WireSourceSpec(
        source_kind="release",
        release_id=references.releases.current_release_id,
        cutoff=None,
        allowed_evidence_kinds=(),
        oracle_entries=(),
    )
    request = helpfulness.FutureChildRequest(
        execution_index=0,
        database_path=str(database_path),
        scope=references.capture.local_scope,
        source=source,
        query=_query(case).decode(),
        future_session_id=f"{case.case_id}-future-session-000",
        future_run_id=f"{case.case_id}-future-run-000",
        renderer_version="memory-codebook/v1",
        consumer_version="scripted-last-occurrence/v1",
    )
    schedule = helpfulness.make_parent_schedule_item(
        execution_index=0,
        case=case,
        references=references,
        arm="current_release",
    )
    return case, references, request, schedule


def test_wire_round_trip_is_compact_sorted_and_exact_typed() -> None:
    request = helpfulness.FutureChildRequest(
        execution_index=7,
        database_path="/tmp/记忆.sqlite3",
        scope=MemoryScope("tenant", "namespace", "subject"),
        source=helpfulness.WireSourceSpec(
            source_kind="raw_evidence",
            release_id=None,
            cutoff=datetime(2026, 7, 9, 1, 2, 3, tzinfo=UTC),
            allowed_evidence_kinds=(
                EvidenceKind.USER_MESSAGE,
                EvidenceKind.FEEDBACK,
            ),
            oracle_entries=(),
        ),
        query="What is the current code for project-abcdef?",
        future_session_id="future-session",
        future_run_id="future-run",
        renderer_version="memory-codebook/v1",
        consumer_version="scripted-last-occurrence/v1",
    )

    encoded = helpfulness.wire_dumps(request)

    assert encoded.endswith("\n")
    assert encoded.count("\n") == 1
    assert ": " not in encoded and ", " not in encoded
    parsed = json.loads(encoded)
    assert (
        encoded
        == json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    decoded = helpfulness.wire_loads(encoded)
    assert type(decoded) is helpfulness.FutureChildRequest
    assert decoded == request
    assert type(decoded.execution_index) is int
    assert type(decoded.scope) is MemoryScope
    assert type(decoded.source.cutoff) is datetime
    assert type(decoded.source.allowed_evidence_kinds) is tuple
    assert all(
        type(kind) is EvidenceKind for kind in decoded.source.allowed_evidence_kinds
    )
    assert type(decoded.source.oracle_entries) is tuple


def test_wire_rejects_unknown_missing_and_wrong_type_fields() -> None:
    request = helpfulness.FutureChildRequest(
        execution_index=7,
        database_path="/tmp/memory.sqlite3",
        scope=MemoryScope("tenant", "namespace", "subject"),
        source=helpfulness.WireSourceSpec(
            source_kind="raw_evidence",
            release_id=None,
            cutoff=datetime(2026, 7, 9, tzinfo=UTC),
            allowed_evidence_kinds=(
                EvidenceKind.USER_MESSAGE,
                EvidenceKind.FEEDBACK,
            ),
            oracle_entries=(),
        ),
        query="query",
        future_session_id="session",
        future_run_id="run",
        renderer_version="renderer",
        consumer_version="consumer",
    )
    valid = json.loads(helpfulness.wire_dumps(request))
    variants = []
    unknown = json.loads(json.dumps(valid))
    unknown["payload"]["expected_response"] = "SECRET"
    variants.append(unknown)
    missing = json.loads(json.dumps(valid))
    del missing["payload"]["query"]
    variants.append(missing)
    bool_index = json.loads(json.dumps(valid))
    bool_index["payload"]["execution_index"] = True
    variants.append(bool_index)
    wrong_scope = json.loads(json.dumps(valid))
    wrong_scope["payload"]["scope"]["tenant_id"] = 3
    variants.append(wrong_scope)
    wrong_datetime = json.loads(json.dumps(valid))
    wrong_datetime["payload"]["source"]["cutoff"] = "not-a-datetime"
    variants.append(wrong_datetime)
    wrong_enum = json.loads(json.dumps(valid))
    wrong_enum["payload"]["source"]["allowed_evidence_kinds"] = ["made_up"]
    variants.append(wrong_enum)
    smuggled_release = json.loads(json.dumps(valid))
    smuggled_release["payload"]["source"]["source_kind"] = "release"
    smuggled_release["payload"]["source"]["release_id"] = "release-id"
    variants.append(smuggled_release)

    for value in variants:
        encoded = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_loads(encoded)
        assert error.value.reason == "closed_schema"


@pytest.mark.parametrize(
    "encoded",
    (
        '{"payload":{},"payload":{},"schema_version":1,"type":"x"}\n',
        '{"payload":{},"schema_version":NaN,"type":"x"}\n',
        '{"payload":{},"schema_version":Infinity,"type":"x"}\n',
        '{"payload":{},"schema_version":1,"type":"x"}\n\n',
    ),
)
def test_wire_rejects_duplicate_keys_nonfinite_numbers_and_extra_framing(
    encoded: str,
) -> None:
    with pytest.raises(helpfulness.WireProtocolError):
        helpfulness.wire_loads(encoded)


def test_wire_normalizes_deep_json_recursion() -> None:
    depth = 10_000
    encoded = (
        '{"payload":'
        + "[" * depth
        + "0"
        + "]" * depth
        + ',"schema_version":1,"type":"capture_child_request"}\n'
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_loads(encoded)
    assert error.value.reason == "closed_schema"


def test_wire_rejects_valid_but_noncanonical_json() -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path="/tmp/capture.sqlite3",
    )
    parsed = json.loads(helpfulness.wire_dumps(request))
    noncanonical = json.dumps(parsed, ensure_ascii=True) + "\n"

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_loads(noncanonical)
    assert error.value.reason == "framing"


def test_wire_rejects_wrong_fixed_tuple_lengths(tmp_path: Path) -> None:
    _case, _path, references, _store = _build_case_graph(tmp_path)
    response = helpfulness.CaptureChildResponse(
        case_index=0,
        references=references,
        pid=123,
        process_instance_id="00000000-0000-4000-8000-000000000000",
        isolated_mode=True,
        areal_module_path=str(
            Path(helpfulness.__file__).resolve().parents[2] / "areal" / "__init__.py"
        ),
        visible_forbidden_environment=(),
        environment_clean=True,
    )
    valid = json.loads(helpfulness.wire_dumps(response))
    variants = []
    short_sessions = json.loads(json.dumps(valid))
    short_sessions["payload"]["references"]["capture"]["capture_session_ids"] = [
        "one",
        "two",
    ]
    variants.append(short_sessions)
    short_controls = json.loads(json.dumps(valid))
    short_controls["payload"]["references"]["capture"]["control_evidence_ids"] = ["one"]
    variants.append(short_controls)

    for value in variants:
        encoded = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_loads(encoded)
        assert error.value.reason == "closed_schema"


def test_wire_normalizes_value_object_validation_failures() -> None:
    request = helpfulness.FutureChildRequest(
        execution_index=0,
        database_path="/tmp/memory.sqlite3",
        scope=MemoryScope("tenant", "namespace", "subject"),
        source=helpfulness.WireSourceSpec(
            source_kind="release",
            release_id="release",
            cutoff=None,
            allowed_evidence_kinds=(),
            oracle_entries=(),
        ),
        query="query",
        future_session_id="session",
        future_run_id="run",
        renderer_version="renderer",
        consumer_version="consumer",
    )
    value = json.loads(helpfulness.wire_dumps(request))
    value["payload"]["scope"]["tenant_id"] = ""
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_loads(encoded)
    assert error.value.reason == "closed_schema"


def test_wire_normalizes_datetime_overflow() -> None:
    request = helpfulness.FutureChildRequest(
        execution_index=0,
        database_path="/tmp/memory.sqlite3",
        scope=MemoryScope("tenant", "namespace", "subject"),
        source=helpfulness.WireSourceSpec(
            source_kind="raw_evidence",
            release_id=None,
            cutoff=datetime(2026, 7, 9, tzinfo=UTC),
            allowed_evidence_kinds=(
                EvidenceKind.USER_MESSAGE,
                EvidenceKind.FEEDBACK,
            ),
            oracle_entries=(),
        ),
        query="query",
        future_session_id="session",
        future_run_id="run",
        renderer_version="renderer",
        consumer_version="consumer",
    )
    value = json.loads(helpfulness.wire_dumps(request))
    value["payload"]["source"]["cutoff"] = "0001-01-01T00:00:00+14:00"
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_loads(encoded)
    assert error.value.reason == "closed_schema"

    overflowing_request = replace(
        request,
        source=replace(
            request.source,
            cutoff=datetime.fromisoformat("0001-01-01T00:00:00+14:00"),
        ),
    )
    with pytest.raises(helpfulness.WireProtocolError) as encode_error:
        helpfulness.wire_dumps(overflowing_request)
    assert encode_error.value.reason == "closed_schema"


def test_scripted_consumer_receipts_actual_history_length() -> None:
    case = helpfulness.generate_case(0)
    rendered = helpfulness.render_context(
        _case_entries(
            case,
            target_value=case.current_value,
            source_kind="oracle",
        )
    )

    result = helpfulness.consume_scripted(
        _query(case),
        rendered.bytes,
        history=(b"capture-only prior turn",),
    )

    assert result.input_receipt.received_history_length == 1


def test_future_wire_contains_no_capture_owned_metadata(tmp_path: Path) -> None:
    _case, _references, request, _schedule = _wire_future_setup(tmp_path)

    assert {field.name for field in fields(request)}.isdisjoint(
        {
            "capture_pid",
            "capture_process_instance_id",
            "capture_session_ids",
            "case_id",
            "expected_response",
        }
    )
    assert "capture_" not in helpfulness.wire_dumps(request)

    response = helpfulness.run_isolated_child(
        request,
        role="future-child",
        timeout_seconds=20,
    )

    assert {field.name for field in fields(response.observation)}.isdisjoint(
        {
            "capture_pid",
            "capture_process_instance_id",
            "capture_session_ids",
            "case_id",
            "expected_response",
        }
    )
    assert "capture_" not in helpfulness.wire_dumps(response)


def test_child_runs_with_isolated_flag_and_checkout_areal_import(
    tmp_path: Path,
) -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "capture.sqlite3"),
    )

    completed = helpfulness.run_isolated_child_raw(
        request,
        role="capture-child",
        timeout_seconds=20,
    )
    response = helpfulness.wire_loads(completed.stdout)

    assert completed.returncode == 0, completed.stderr
    assert completed.args == [
        sys.executable,
        "-I",
        str(Path(helpfulness.__file__).resolve()),
        "capture-child",
    ]
    assert response.isolated_mode is True
    assert (
        Path(response.areal_module_path)
        .resolve()
        .is_relative_to(Path(helpfulness.__file__).resolve().parents[2])
    )
    assert response.pid != os.getpid()
    assert response.pid == completed.pid
    assert response.process_instance_id != helpfulness.PROCESS_INSTANCE_ID


def test_child_environment_drops_pythonpath_pythonhome_and_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "PYTHONPATH": "/tmp/injected",
        "PYTHONHOME": "/tmp/fake-home",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AZURE_OPENAI_API_KEY": "secret",
        "GOOGLE_API_KEY": "secret",
        "OPENAI_API_KEY": "secret",
        "ANTHROPIC_API_KEY": "secret",
        "HF_TOKEN": "secret",
        "GITHUB_TOKEN": "secret",
        "GH_TOKEN": "secret",
        "AREAL_TASK6_UNKNOWN_CANARY": "must-not-cross-exec",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "capture.sqlite3"),
    )

    response = helpfulness.run_isolated_child(
        request,
        role="capture-child",
        timeout_seconds=20,
    )

    assert response.visible_forbidden_environment == ()
    assert response.environment_clean is True


def test_child_stdout_contains_one_json_envelope(tmp_path: Path) -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "capture.sqlite3"),
    )

    completed = helpfulness.run_isolated_child_raw(
        request,
        role="capture-child",
        timeout_seconds=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("\n") == 1
    assert completed.stdout.endswith("\n")
    assert type(completed.stderr) is str
    response = helpfulness.wire_loads(completed.stdout)
    assert type(response) is helpfulness.CaptureChildResponse
    assert completed.stdout == helpfulness.wire_dumps(response)


def test_capture_launcher_rejects_forged_process_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "capture.sqlite3"),
    )
    completed = helpfulness.run_isolated_child_raw(
        request,
        role="capture-child",
        timeout_seconds=20,
    )
    response = helpfulness.wire_loads(completed.stdout)
    assert type(response) is helpfulness.CaptureChildResponse
    mutants = (
        replace(response, isolated_mode=False),
        replace(
            response,
            visible_forbidden_environment=("OPENAI_API_KEY",),
            environment_clean=False,
        ),
        replace(response, process_instance_id=""),
        replace(response, process_instance_id="00000000-0000-0000-0000-000000000000"),
        replace(response, areal_module_path="/tmp/foreign-checkout/areal/__init__.py"),
    )

    for mutant in mutants:
        forged = replace(completed, stdout=helpfulness.wire_dumps(mutant))
        monkeypatch.setattr(
            helpfulness,
            "run_isolated_child_raw",
            lambda *_args, _forged=forged, **_kwargs: _forged,
        )
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness.run_isolated_child(
                request,
                role="capture-child",
                timeout_seconds=20,
            )
        assert error.value.reason == "process_isolation"


def test_parent_rejects_nonzero_exit_even_with_valid_child_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "capture.sqlite3"),
    )
    completed = helpfulness.run_isolated_child_raw(
        request,
        role="capture-child",
        timeout_seconds=20,
    )
    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        lambda *_args, **_kwargs: replace(completed, returncode=9),
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child(
            request,
            role="capture-child",
            timeout_seconds=20,
        )
    assert error.value.reason == "child_nonzero_exit"


def test_parent_rejects_extra_stdout_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "capture.sqlite3"),
    )
    completed = helpfulness.run_isolated_child_raw(
        request,
        role="capture-child",
        timeout_seconds=20,
    )
    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        lambda *_args, **_kwargs: replace(
            completed,
            stdout=completed.stdout + "unexpected",
        ),
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child(
            request,
            role="capture-child",
            timeout_seconds=20,
        )
    assert error.value.reason == "framing"


def test_child_timeout_has_stable_protocol_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.returncode = -9
            self.communicate_calls = 0
            self.killed = False
            self.reaped = False

        def communicate(self, *, input=None, timeout=None):
            self.communicate_calls += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired("child", timeout)
            self.reaped = True
            return b"", b""

        def kill(self) -> None:
            self.killed = True

    process = HangingProcess()
    monkeypatch.setattr(
        helpfulness.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "capture.sqlite3"),
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child_raw(
            request,
            role="capture-child",
            timeout_seconds=0.01,
        )
    assert error.value.reason == "child_timeout"
    assert process.communicate_calls == 2
    assert process.killed is True
    assert process.reaped is True


def test_ground_truth_child_payload_is_rejected(tmp_path: Path) -> None:
    _case, _references, request, _schedule = _wire_future_setup(tmp_path)
    value = json.loads(helpfulness.wire_dumps(request))
    value["payload"]["expected_response"] = "LEAKED-TRUTH"
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_loads(encoded)
    assert error.value.reason == "closed_schema"


def test_missing_local_release_is_not_misclassified_as_foreign_scope(
    tmp_path: Path,
) -> None:
    _case, _references, request, _schedule = _wire_future_setup(tmp_path)
    missing = replace(
        request,
        source=replace(request.source, release_id="missing-local-release"),
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child(
            missing,
            role="future-child",
            timeout_seconds=20,
        )
    assert error.value.reason == "child_nonzero_exit"


@pytest.mark.parametrize(
    ("mutant", "reason"),
    (
        ("in_process", "process_isolation"),
        ("identity", "process_isolation"),
        ("history", "history_nonzero"),
        ("assignment", "assignment_mismatch"),
        ("foreign", "foreign_scope"),
    ),
)
def test_child_execution_mutants_fail_for_pre_registered_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutant: str,
    reason: str,
) -> None:
    case, references, request, schedule = _wire_future_setup(tmp_path)
    if mutant == "in_process":
        response = helpfulness.execute_future_child_request(request)
    elif mutant == "foreign":
        foreign_request = replace(
            request,
            source=replace(
                request.source,
                release_id=references.releases.foreign_sentinel_release_id,
            ),
        )
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness.run_isolated_child(
                foreign_request,
                role="future-child",
                timeout_seconds=20,
            )
        assert error.value.reason == reason
        return
    else:
        response = helpfulness.run_isolated_child(
            request,
            role="future-child",
            timeout_seconds=20,
        )
        if mutant == "identity":
            response = replace(
                response,
                process_instance_id="",
                observation=replace(
                    response.observation,
                    future_process_instance_id="",
                ),
            )
        elif mutant == "history":
            isolated_response = response
            consume_without_leak = helpfulness.consume_scripted

            def consume_with_hidden_history(query, rendered_context, *, history=()):
                return consume_without_leak(
                    query,
                    rendered_context,
                    history=(b"capture-only prior turn",),
                )

            monkeypatch.setattr(
                helpfulness,
                "consume_scripted",
                consume_with_hidden_history,
            )
            leaky_response = helpfulness.execute_future_child_request(request)
            assert leaky_response.observation.history_length == 1
            assert (
                leaky_response.observation.consumer_input_receipt.received_history_length
                == 1
            )
            response = replace(
                isolated_response,
                observation=replace(
                    leaky_response.observation,
                    future_pid=isolated_response.pid,
                    future_process_instance_id=(isolated_response.process_instance_id),
                ),
            )
        elif mutant == "assignment":
            schedule = helpfulness.make_parent_schedule_item(
                execution_index=request.execution_index,
                case=case,
                references=references,
                arm="stale_release",
            )

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.validate_future_child_response(response, schedule)
    assert error.value.reason == reason
