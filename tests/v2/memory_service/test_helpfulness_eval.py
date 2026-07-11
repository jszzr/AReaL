# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic Memory Service helpfulness evaluator."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Callable
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

MODEL_ARMS = (
    "current_release",
    "raw_history",
    "memory_off",
    "target_masked",
    "stale_release",
    "oracle",
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


def _guard_source_render(monkeypatch: pytest.MonkeyPatch) -> Counter:
    calls = Counter()

    def forbidden(stage: str):
        def fail(*_args, **_kwargs):
            calls[stage] += 1
            raise AssertionError(f"{stage} ran for an invalid mutation")

        return fail

    monkeypatch.setattr(
        helpfulness,
        "render_context",
        forbidden("render"),
    )
    return calls


def _guard_trace_output(monkeypatch: pytest.MonkeyPatch) -> Counter:
    calls = Counter()

    def trace_must_not_run(*_args, **_kwargs):
        calls["trace"] += 1
        raise AssertionError("trace ran for an invalid mutation")

    monkeypatch.setattr(
        helpfulness,
        "_trace_from_observation",
        trace_must_not_run,
    )
    return calls


def _guard_process_join(monkeypatch: pytest.MonkeyPatch) -> Counter:
    calls = Counter()

    def forbidden(stage: str):
        def fail(*_args, **_kwargs):
            calls[stage] += 1
            raise AssertionError(f"{stage} ran for an invalid process mutation")

        return fail

    monkeypatch.setattr(
        helpfulness,
        "parent_join_and_score",
        forbidden("join"),
    )
    return calls


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
def test_source_and_provenance_mutation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    downstream_calls = _guard_source_render(monkeypatch)
    with pytest.raises(helpfulness.TreatmentValidationError) as error:
        treatment = helpfulness.resolve_and_validate(
            resolver,
            capability,
            database_path=path,
            assignment=assignment,
            audit_sink=audit,
        )
        helpfulness.render_context(treatment.entries)
    assert error.value.reason == reason
    assert downstream_calls == Counter()


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


def test_ids_are_not_injected_before_consumer_receipt_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    invalid = replace(
        observation,
        consumer_input_receipt=replace(
            observation.consumer_input_receipt,
            received_context_sha256="0" * 64,
        ),
    )
    trace_calls = _guard_trace_output(monkeypatch)
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (invalid,),
            (bundle["schedule"],),
            enforce_scripted_outcomes=False,
        )
    assert error.value.reason == "received_context_mismatch"
    assert trace_calls == Counter()


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
            (observations[0], observations[0], observations[1]),
            "execution_index_mismatch",
        ),
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

    duplicate_schedule = (schedule[0], schedule[0], schedule[1])
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            observations,
            duplicate_schedule,
            enforce_scripted_outcomes=False,
        )
    assert error.value.reason == "execution_index_mismatch"


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
def test_exposure_and_consumer_mutation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    trace_calls = _guard_trace_output(monkeypatch)
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness.parent_join_and_score(
            (observation,),
            (bundle["schedule"],),
            enforce_scripted_outcomes=True,
        )
    assert error.value.reason == reason
    assert trace_calls == Counter()


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


def test_wire_rejects_strings_that_are_not_strict_utf8(
    tmp_path: Path,
) -> None:
    _case, _references, request, _schedule = _wire_future_setup(tmp_path)
    invalid = replace(request, query=f"{request.query}\ud800")

    with pytest.raises(helpfulness.WireProtocolError) as encode_error:
        helpfulness.wire_dumps(invalid)
    assert encode_error.value.reason == "closed_schema"

    value = json.loads(helpfulness.wire_dumps(request))
    value["payload"]["query"] = f"{request.query}\ud800"
    encoded = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with pytest.raises(helpfulness.WireProtocolError) as decode_error:
        helpfulness.wire_loads(encoded)
    assert decode_error.value.reason == "closed_schema"


def test_run_raw_rejects_non_utf8_wire_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _case, _references, request, _schedule = _wire_future_setup(tmp_path)
    invalid = replace(request, query=f"{request.query}\ud800")
    calls = Counter()

    def forbidden_popen(*_args, **_kwargs):
        calls["popen"] += 1
        raise AssertionError("Popen must not run for an invalid wire string")

    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden_popen)
    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child_raw(
            invalid,
            role="future-child",
            timeout_seconds=20,
        )

    assert error.value.reason == "closed_schema"
    assert calls == Counter()


def test_wire_canonically_round_trips_valid_non_ascii_strings() -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path="/tmp/记忆服务.sqlite3",
    )

    encoded = helpfulness.wire_dumps(request)

    assert "记忆服务" in encoded
    assert helpfulness.wire_loads(encoded) == request
    assert helpfulness.wire_dumps(helpfulness.wire_loads(encoded)) == encoded


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


def test_child_response_rejects_missing_process_identity(
    tmp_path: Path,
) -> None:
    _case, _references, request, schedule = _wire_future_setup(tmp_path)
    response = helpfulness.run_isolated_child(
        request,
        role="future-child",
        timeout_seconds=20,
    )
    response = replace(
        response,
        process_instance_id="",
        observation=replace(
            response.observation,
            future_process_instance_id="",
        ),
    )

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.validate_future_child_response(response, schedule)
    assert error.value.reason == "process_isolation"


def test_single_future_cannot_classify_a_foreign_sentinel_from_local_not_found(
    tmp_path: Path,
) -> None:
    _case, references, request, _schedule = _wire_future_setup(tmp_path)
    foreign_request = replace(
        request,
        source=replace(
            request.source,
            release_id=references.releases.foreign_sentinel_release_id,
        ),
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child(
            foreign_request,
            role="future-child",
            timeout_seconds=20,
        )

    assert error.value.reason == "child_nonzero_exit"


@pytest.fixture(scope="module")
def fast_profile_bundle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("fast-profile")
    artifact_path = root / "replay.jsonl"
    original_popen = helpfulness.subprocess.Popen
    original_store = helpfulness.SQLiteMemoryStore
    launches = []

    def spy_popen(*args, **kwargs):
        launches.append(tuple(args[0]))
        return original_popen(*args, **kwargs)

    def parent_store_access_is_forbidden(*_args, **_kwargs):
        raise AssertionError("the parent opened a case database")

    helpfulness.subprocess.Popen = spy_popen
    helpfulness.SQLiteMemoryStore = parent_store_access_is_forbidden
    try:
        execution = helpfulness._execute_fast_profile_children(
            root / "databases",
            timeout_seconds=120,
        )
        result = helpfulness._finalize_fast_profile_execution(
            execution,
            artifact_path=artifact_path,
        )
    finally:
        helpfulness.subprocess.Popen = original_popen
        helpfulness.SQLiteMemoryStore = original_store
    return {
        "artifact_path": artifact_path,
        "execution": execution,
        "launches": tuple(launches),
        "result": result,
        "root": root,
    }


@pytest.mark.parametrize(
    ("mutation", "error_type", "reason"),
    (
        (
            "ground_truth",
            helpfulness.WireProtocolError,
            "closed_schema",
        ),
        (
            "in_process",
            helpfulness.ChildExecutionValidationError,
            "process_isolation",
        ),
        (
            "history_leak",
            helpfulness.ChildExecutionValidationError,
            "history_nonzero",
        ),
        (
            "assignment_swap",
            helpfulness.ChildExecutionValidationError,
            "assignment_mismatch",
        ),
        (
            "foreign_scope",
            helpfulness.ChildExecutionValidationError,
            "foreign_scope",
        ),
    ),
)
def test_process_assignment_and_leakage_mutation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_type: type[Exception],
    reason: str,
) -> None:
    case, references, request, schedule = _wire_future_setup(tmp_path)
    derived_calls = _guard_process_join(monkeypatch)

    def reach_derived_pipeline() -> None:
        helpfulness.parent_join_and_score(
            (),
            (),
            enforce_scripted_outcomes=False,
        )

    if mutation == "ground_truth":
        value = json.loads(helpfulness.wire_dumps(request))
        value["payload"]["expected_response"] = case.current_value
        encoded = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        with pytest.raises(error_type) as error:
            helpfulness.wire_loads(encoded)
            reach_derived_pipeline()
    elif mutation == "in_process":
        response = helpfulness.execute_future_child_request(request)
        with pytest.raises(error_type) as error:
            helpfulness.validate_future_child_response(response, schedule)
            reach_derived_pipeline()
    elif mutation == "history_leak":
        isolated = helpfulness.run_isolated_child(
            request,
            role="future-child",
            timeout_seconds=20,
        )
        original_consumer = helpfulness.consume_scripted

        def consume_with_history(query, rendered_context, *, history=()):
            return original_consumer(
                query,
                rendered_context,
                history=(b"capture-only prior turn",),
            )

        monkeypatch.setattr(
            helpfulness,
            "consume_scripted",
            consume_with_history,
        )
        leaky = helpfulness.execute_future_child_request(request)
        response = replace(
            isolated,
            observation=replace(
                leaky.observation,
                future_pid=isolated.pid,
                future_process_instance_id=isolated.process_instance_id,
            ),
        )
        with pytest.raises(error_type) as error:
            helpfulness.validate_future_child_response(response, schedule)
            reach_derived_pipeline()
    elif mutation == "assignment_swap":
        response = helpfulness.run_isolated_child(
            request,
            role="future-child",
            timeout_seconds=20,
        )
        stale_schedule = helpfulness.make_parent_schedule_item(
            execution_index=request.execution_index,
            case=case,
            references=references,
            arm="stale_release",
        )
        with pytest.raises(error_type) as error:
            helpfulness.validate_future_child_response(response, stale_schedule)
            reach_derived_pipeline()
    else:
        foreign_request = replace(
            request,
            scope=references.capture.foreign_scope,
            source=replace(
                request.source,
                release_id=references.releases.foreign_sentinel_release_id,
            ),
        )
        response = helpfulness.run_isolated_child(
            foreign_request,
            role="future-child",
            timeout_seconds=20,
        )
        with pytest.raises(error_type) as error:
            helpfulness.validate_future_child_response(response, schedule)
            reach_derived_pipeline()

    assert error.value.reason == reason
    assert derived_calls == Counter()


def test_fast_profile_produces_48_outcomes_and_8_foreign_probes(
    fast_profile_bundle,
) -> None:
    result = fast_profile_bundle["result"]
    launches = fast_profile_bundle["launches"]
    execution = fast_profile_bundle["execution"]

    script_path = str(Path(helpfulness.__file__).resolve())
    assert launches == (
        (sys.executable, "-I", script_path, "capture-child"),
        (sys.executable, "-I", script_path, "future-child"),
    )
    assert (
        execution.capture_response.process_instance_id
        != helpfulness.PROCESS_INSTANCE_ID
    )
    assert (
        execution.future_response.process_instance_id != helpfulness.PROCESS_INSTANCE_ID
    )
    assert len(result.outcomes) == 48
    assert len(result.foreign_probes) == 8
    assert {trace.execution_index for trace in result.outcomes} == set(range(48))
    assert {trace.execution_index for trace in result.foreign_probes} == set(
        range(48, 56)
    )


def test_batch_not_found_witnesses_read_only_the_requested_local_scope(
    fast_profile_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = fast_profile_bundle["execution"]
    foreign_release_ids = {
        item.references.releases.foreign_sentinel_release_id
        for item in execution.capture_response.items
    }
    probe_items = tuple(
        item
        for item in execution.future_request.items
        if item.source.release_id in foreign_release_ids
    )
    expected_scopes = {item.scope for item in probe_items}
    original_get_release = helpfulness.SQLiteMemoryStore.get_release
    reads = []

    def spy_get_release(store, scope, release_id):
        reads.append((scope, release_id))
        return original_get_release(store, scope, release_id)

    monkeypatch.setattr(
        helpfulness.SQLiteMemoryStore,
        "get_release",
        spy_get_release,
    )
    response = helpfulness.execute_future_batch_request(
        helpfulness.FutureBatchRequest(items=probe_items)
    )

    assert len(probe_items) == 8
    assert len(response.foreign_probes) == 8
    assert {probe.reason for probe in response.foreign_probes} == {"release_not_found"}
    assert len(reads) == 8
    assert {scope for scope, _release_id in reads} == expected_scopes
    assert all(not scope.subject_id.endswith("-foreign") for scope, _ in reads)


def test_fast_profile_future_batch_has_no_explicit_scorer_metadata(
    fast_profile_bundle,
) -> None:
    """Check the honest-runner boundary, not secrecy from a malicious child."""

    execution = fast_profile_bundle["execution"]
    encoded = helpfulness.wire_dumps(execution.future_request)
    payload = json.loads(encoded)["payload"]

    assert set(payload) == {"items"}
    assert len(payload["items"]) == 56
    assert all(
        set(item)
        == {
            "consumer_version",
            "database_path",
            "execution_index",
            "future_run_id",
            "future_session_id",
            "query",
            "renderer_version",
            "scope",
            "source",
        }
        for item in payload["items"]
    )
    assert all(
        forbidden not in encoded
        for forbidden in (
            '"arm"',
            '"case_id"',
            '"capture_',
            '"expected_response"',
            '"utility"',
        )
    )


def test_fast_future_wire_indexes_are_opaque_and_not_parent_logical_indexes(
    fast_profile_bundle,
) -> None:
    execution = fast_profile_bundle["execution"]
    round_tripped = helpfulness.wire_loads(
        helpfulness.wire_dumps(execution.future_request)
    )
    assert type(round_tripped) is helpfulness.FutureBatchRequest
    bindings = execution.execution_bindings
    logical_by_opaque = {
        binding.opaque_execution_index: binding.logical_execution_index
        for binding in bindings
    }
    wire_tokens = tuple(item.execution_index for item in round_tripped.items)
    logical_order = tuple(logical_by_opaque[token] for token in wire_tokens)

    assert len(bindings) == 56
    assert set(logical_by_opaque.values()) == set(range(56))
    assert wire_tokens == tuple(sorted(wire_tokens))
    assert len(set(wire_tokens)) == 56
    assert all(token.bit_length() == 128 for token in wire_tokens)
    assert set(wire_tokens).isdisjoint(range(56))
    assert logical_order != tuple(range(56))
    assert any(logical >= 48 for logical in logical_order[:48])
    assert any(logical < 48 for logical in logical_order[48:])
    for remainder in range(6):
        possible_arms = {
            logical % 6
            for token, logical in logical_by_opaque.items()
            if logical < 48 and token % 6 == remainder
        }
        assert len(possible_arms) >= 2
    for item in round_tripped.items:
        suffix = f"{item.execution_index:032x}"
        assert item.future_session_id == f"future-session-{suffix}"
        assert item.future_run_id == f"future-run-{suffix}"
        logical = logical_by_opaque[item.execution_index]
        assert item.future_session_id != f"future-session-{logical:03d}"
        assert item.future_run_id != f"future-run-{logical:03d}"

    response_indexes = {
        item.execution_index for item in execution.future_response.observations
    } | {item.execution_index for item in execution.future_response.foreign_probes}
    receipt_indexes = {
        item.execution_index for item in execution.future_response.state_receipts
    }
    assert response_indexes == set(wire_tokens)
    assert receipt_indexes == set(wire_tokens)


def test_fast_opaque_execution_token_rejects_negative_128_bit_magnitude() -> None:
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._opaque_future_identity(-(1 << 127))

    assert error.value.reason == "assignment_mismatch"


def test_fast_profile_matches_all_eight_strict_signatures(
    fast_profile_bundle,
) -> None:
    result = fast_profile_bundle["result"]

    assert len(result.signatures) == 8
    for signature in result.signatures:
        case = helpfulness.generate_case(signature.case_index)
        assert signature.case_id == case.case_id
        assert signature.normalized_responses == (
            case.current_value,
            case.current_value,
            "UNKNOWN",
            "UNKNOWN",
            case.old_value,
            case.current_value,
        )
        assert signature.matches is True
        case_traces = tuple(
            trace for trace in result.outcomes if trace.case_id == case.case_id
        )
        by_arm = {trace.arm: trace for trace in case_traces}
        assert (
            by_arm["current_release"].rendered_context_sha256
            == by_arm["oracle"].rendered_context_sha256
        )
        assert (
            by_arm["current_release"].rendered_context_utf8_bytes
            == by_arm["oracle"].rendered_context_utf8_bytes
        )


def test_fast_profile_uses_distinct_capture_and_future_instances(
    fast_profile_bundle,
) -> None:
    result = fast_profile_bundle["result"]
    traces = (*result.outcomes, *result.foreign_probes)

    capture_instances = {trace.capture_process_instance_id for trace in traces}
    future_instances = {trace.future_process_instance_id for trace in traces}
    assert len(capture_instances) == 1
    assert len(future_instances) == 1
    assert capture_instances.isdisjoint(future_instances)
    assert len({trace.capture_pid for trace in traces}) == 1
    assert len({trace.future_pid for trace in traces}) == 1
    assert len({trace.future_session_id for trace in traces}) == 56
    assert len({trace.future_run_id for trace in traces}) == 56


def test_fast_profile_recreates_every_per_item_state(
    fast_profile_bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = fast_profile_bundle["result"]
    receipts = result.state_receipts
    identity_fields = (
        "store_instance_id",
        "reader_instance_id",
        "resolver_instance_id",
        "renderer_instance_id",
        "consumer_instance_id",
        "audit_instance_id",
        "logical_session_instance_id",
        "history_instance_id",
    )

    assert len(receipts) == 56
    assert tuple(receipt.generation_index for receipt in receipts) == tuple(range(56))
    assert all(receipt.history_length == 0 for receipt in receipts)
    for field_name in identity_fields:
        assert len({getattr(receipt, field_name) for receipt in receipts}) == 56

    _case, _references, request, _schedule = _wire_future_setup(tmp_path)
    batch = helpfulness.FutureBatchRequest(
        items=(
            request,
            replace(
                request,
                execution_index=1,
                future_session_id="future-session-001",
                future_run_id="future-run-001",
            ),
        )
    )
    original_factory = helpfulness._new_item_execution_state
    constructed = []

    def spy_factory(item, generation_index):
        state = original_factory(item, generation_index)
        constructed.append(state)
        return state

    monkeypatch.setattr(helpfulness, "_new_item_execution_state", spy_factory)
    helpfulness.execute_future_batch_request(batch)
    assert len(constructed) == 2
    for field_name in (
        "store",
        "reader",
        "resolver",
        "renderer",
        "consumer",
        "audit",
        "logical_session",
        "history",
    ):
        assert len({id(getattr(state, field_name)) for state in constructed}) == 2

    reused = constructed[0]
    monkeypatch.setattr(
        helpfulness,
        "_new_item_execution_state",
        lambda _item, _generation_index: reused,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_future_batch_request(batch)
    assert error.value.reason == "state_reuse"

    shared_store = constructed[0].store

    def shared_store_factory(item, generation_index):
        state = original_factory(item, generation_index)
        state.store = shared_store
        return state

    monkeypatch.setattr(
        helpfulness,
        "_new_item_execution_state",
        shared_store_factory,
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_future_batch_request(batch)
    assert error.value.reason == "state_reuse"


@pytest.mark.parametrize("version_field", ["renderer_version", "consumer_version"])
def test_future_batch_prevalidates_every_item_before_creating_any_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_field: str,
) -> None:
    _case, _references, request, _schedule = _wire_future_setup(tmp_path)
    bad_last = replace(
        request,
        execution_index=1,
        future_session_id="future-session-bad-last",
        future_run_id="future-run-bad-last",
        **{version_field: "unsupported-version/v999"},
    )
    batch = helpfulness.FutureBatchRequest(items=(request, bad_last))
    original_factory = helpfulness._new_item_execution_state
    original_consumer = helpfulness.consume_scripted
    calls = Counter()

    def spy_factory(item, generation_index):
        calls["state"] += 1
        return original_factory(item, generation_index)

    def spy_consumer(query, context, *, history=()):
        calls["consumer"] += 1
        return original_consumer(query, context, history=history)

    monkeypatch.setattr(helpfulness, "_new_item_execution_state", spy_factory)
    monkeypatch.setattr(helpfulness, "consume_scripted", spy_consumer)

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.execute_future_batch_request(batch)

    assert error.value.reason == "assignment_mismatch"
    assert calls == Counter()


@pytest.mark.parametrize(
    ("bad_query", "error_type", "reason"),
    [
        (
            "What is the code?",
            helpfulness.ChildExecutionValidationError,
            "assignment_mismatch",
        ),
        (
            "What is the current code for project-abc234?\ud800",
            helpfulness.WireProtocolError,
            "closed_schema",
        ),
    ],
)
def test_future_batch_prevalidates_strict_query_before_store_or_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_query: str,
    error_type: type[Exception],
    reason: str,
) -> None:
    _case, _references, request, _schedule = _wire_future_setup(tmp_path)
    bad_last = replace(
        request,
        execution_index=1,
        future_session_id="future-session-bad-query",
        future_run_id="future-run-bad-query",
        query=bad_query,
    )
    batch = helpfulness.FutureBatchRequest(items=(request, bad_last))
    original_store = helpfulness.SQLiteMemoryStore
    original_factory = helpfulness._new_item_execution_state
    original_consumer = helpfulness.consume_scripted
    calls = Counter()

    def spy_store(*args, **kwargs):
        calls["store"] += 1
        return original_store(*args, **kwargs)

    def spy_factory(item, generation_index):
        calls["state"] += 1
        return original_factory(item, generation_index)

    def spy_consumer(query, context, *, history=()):
        calls["consumer"] += 1
        return original_consumer(query, context, history=history)

    monkeypatch.setattr(helpfulness, "SQLiteMemoryStore", spy_store)
    monkeypatch.setattr(helpfulness, "_new_item_execution_state", spy_factory)
    monkeypatch.setattr(helpfulness, "consume_scripted", spy_consumer)

    with pytest.raises(error_type) as error:
        helpfulness.execute_future_batch_request(batch)

    assert error.value.reason == reason
    assert calls == Counter()


def test_fast_profile_batch_join_is_order_independent_but_complete(
    fast_profile_bundle,
) -> None:
    execution = fast_profile_bundle["execution"]
    response = execution.future_response
    reversed_execution = replace(
        execution,
        future_response=replace(
            response,
            observations=tuple(reversed(response.observations)),
            foreign_probes=tuple(reversed(response.foreign_probes)),
        ),
    )

    reversed_result = helpfulness._finalize_fast_profile_execution(
        reversed_execution,
        artifact_path=None,
    )
    assert reversed_result.outcomes == fast_profile_bundle["result"].outcomes
    assert (
        reversed_result.foreign_probes == fast_profile_bundle["result"].foreign_probes
    )

    duplicate_execution = replace(
        execution,
        future_response=replace(
            response,
            observations=(*response.observations, response.observations[0]),
        ),
    )
    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness._finalize_fast_profile_execution(
            duplicate_execution,
            artifact_path=None,
        )
    assert error.value.reason == "execution_index_mismatch"

    first, second, *rest = response.observations
    swapped_execution = replace(
        execution,
        future_response=replace(
            response,
            observations=(
                replace(first, execution_index=second.execution_index),
                replace(second, execution_index=first.execution_index),
                *rest,
            ),
        ),
    )
    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._finalize_fast_profile_execution(
            swapped_execution,
            artifact_path=None,
        )
    assert error.value.reason == "assignment_mismatch"


def test_fast_profile_parent_independently_rejects_forged_foreign_contract(
    fast_profile_bundle,
) -> None:
    execution = fast_profile_bundle["execution"]
    capture = execution.capture_response
    future_request = execution.future_request
    future_response = execution.future_response
    first_item = capture.items[0]
    references = first_item.references

    def same_length_forgery(value: str) -> str:
        replacement = "0" if value[-1] != "0" else "1"
        return value[:-1] + replacement

    forged_release_id = same_length_forgery(
        references.releases.foreign_sentinel_release_id
    )
    forged_references = replace(
        references,
        capture=replace(
            references.capture,
            foreign_scope=replace(
                references.capture.foreign_scope,
                subject_id=same_length_forgery(
                    references.capture.foreign_scope.subject_id
                ),
            ),
            foreign_evidence_id=same_length_forgery(
                references.capture.foreign_evidence_id
            ),
        ),
        revisions=replace(
            references.revisions,
            foreign_target_revision_id=same_length_forgery(
                references.revisions.foreign_target_revision_id
            ),
        ),
        releases=replace(
            references.releases,
            foreign_sentinel_release_id=forged_release_id,
        ),
    )
    forged_capture = replace(
        capture,
        items=(replace(first_item, references=forged_references), *capture.items[1:]),
    )
    probe_token = next(
        binding.opaque_execution_index
        for binding in execution.execution_bindings
        if binding.logical_execution_index == 48
    )
    probe_request_position = next(
        index
        for index, item in enumerate(future_request.items)
        if item.execution_index == probe_token
    )
    probe_request = future_request.items[probe_request_position]
    forged_future_request = replace(
        future_request,
        items=tuple(
            replace(
                probe_request,
                source=replace(probe_request.source, release_id=forged_release_id),
            )
            if index == probe_request_position
            else item
            for index, item in enumerate(future_request.items)
        ),
    )
    probe_response_position = next(
        index
        for index, probe in enumerate(future_response.foreign_probes)
        if probe.execution_index == probe_token
    )
    first_probe = future_response.foreign_probes[probe_response_position]
    forged_future_response = replace(
        future_response,
        foreign_probes=tuple(
            replace(first_probe, release_id=forged_release_id)
            if index == probe_response_position
            else probe
            for index, probe in enumerate(future_response.foreign_probes)
        ),
    )
    forged_execution = replace(
        execution,
        capture_response=forged_capture,
        future_request=forged_future_request,
        future_response=forged_future_response,
    )

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._finalize_fast_profile_execution(
            forged_execution,
            artifact_path=None,
        )
    assert error.value.reason == "foreign_scope"


def test_fast_profile_writes_and_strictly_replays_canonical_jsonl(
    fast_profile_bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = fast_profile_bundle["artifact_path"]
    result = fast_profile_bundle["result"]
    lines = artifact_path.read_text(encoding="utf-8").splitlines(keepends=True)

    assert len(lines) == 57
    decoded = tuple(helpfulness.wire_loads(line) for line in lines)
    assert type(decoded[0]) is helpfulness.ReplayHeader
    assert all(type(item) is helpfulness.EvaluationTrace for item in decoded[1:49])
    assert all(type(item) is helpfulness.LeakageSentinelTrace for item in decoded[49:])
    assert all(
        helpfulness.wire_dumps(item) == line
        for item, line in zip(decoded, lines, strict=True)
    )

    replay = helpfulness.read_fast_profile_artifact(artifact_path)
    assert type(replay) is helpfulness.ReplayedFastRun
    assert not isinstance(replay, helpfulness.FastProfileResult)
    assert replay.header == decoded[0]
    assert replay.outcomes == result.outcomes
    assert replay.foreign_probes == result.foreign_probes
    assert replay.signatures == result.signatures
    assert replay.header.case_manifest_sha256s == FIRST_EIGHT_MANIFEST_SHA256

    untouched = tmp_path / "none-means-no-artifact.jsonl"

    def unexpected_writer(*_args, **_kwargs):
        raise AssertionError("artifact writer ran for artifact_path=None")

    monkeypatch.setattr(
        helpfulness,
        "_write_fast_profile_artifact",
        unexpected_writer,
    )
    helpfulness._finalize_fast_profile_execution(
        fast_profile_bundle["execution"],
        artifact_path=None,
    )
    assert not untouched.exists()


def test_fast_profile_replay_is_file_only_and_cannot_claim_live_provenance(
    fast_profile_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_live_operation(*_args, **_kwargs):
        raise AssertionError("replay touched a live process or database")

    monkeypatch.setattr(helpfulness, "SQLiteMemoryStore", forbidden_live_operation)
    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden_live_operation)

    replay = helpfulness.read_fast_profile_artifact(
        fast_profile_bundle["artifact_path"]
    )
    assert type(replay) is helpfulness.ReplayedFastRun
    assert not hasattr(replay, "state_receipts")

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness._finalize_fast_profile_execution(
            replay,
            artifact_path=None,
        )
    assert error.value.reason == "replay_provenance"


def test_fast_profile_replay_rejects_truncation_duplicates_extras_and_type_swaps(
    fast_profile_bundle,
    tmp_path: Path,
) -> None:
    original = (
        fast_profile_bundle["artifact_path"]
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    first_outcome = helpfulness.wire_loads(original[1])
    first_probe = helpfulness.wire_loads(original[49])
    assert type(first_outcome) is helpfulness.EvaluationTrace
    assert type(first_probe) is helpfulness.LeakageSentinelTrace
    case_one = helpfulness.generate_case(1)
    forged_manifest = helpfulness.wire_dumps(
        replace(
            first_outcome,
            case_manifest_sha256=FIRST_EIGHT_MANIFEST_SHA256[1],
        )
    )
    forged_probe_case = helpfulness.wire_dumps(
        replace(
            first_probe,
            case_id=case_one.case_id,
            case_manifest_sha256=FIRST_EIGHT_MANIFEST_SHA256[1],
            requested_scope=MemoryScope(
                "memory-eval",
                "scoped-codebook-v1",
                case_one.subject_id,
            ),
            companion_scope=MemoryScope(
                "memory-eval",
                "scoped-codebook-v1",
                f"{case_one.subject_id}-foreign",
            ),
        )
    )

    def alter_last_character(value: str) -> str:
        return value[:-1] + ("0" if value[-1] != "0" else "1")

    forged_probe_release = helpfulness.wire_dumps(
        replace(
            first_probe,
            foreign_release_id=alter_last_character(first_probe.foreign_release_id),
        )
    )
    forged_probe_evidence = helpfulness.wire_dumps(
        replace(
            first_probe,
            foreign_evidence_id=alter_last_character(first_probe.foreign_evidence_id),
        )
    )
    forged_probe_session = helpfulness.wire_dumps(
        replace(first_probe, future_session_id="future-session-049")
    )
    forged_probe_run = helpfulness.wire_dumps(
        replace(first_probe, future_run_id="future-run-049")
    )
    forged_probe_history = helpfulness.wire_dumps(
        replace(first_probe, history_length=1)
    )
    forged_probe_reason_value = json.loads(original[49])
    forged_probe_reason_value["payload"]["reason"] = "accepted_foreign_release"
    forged_probe_reason = (
        json.dumps(
            forged_probe_reason_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    corruptions = {
        "truncated": original[:-1],
        "duplicate": [*original[:2], original[1], *original[2:]],
        "extra": [*original, original[0]],
        "manifest-swap": [original[0], forged_manifest, *original[2:]],
        "probe-case-swap": [*original[:49], forged_probe_case, *original[50:]],
        "probe-release": [
            *original[:49],
            forged_probe_release,
            *original[50:],
        ],
        "probe-evidence": [
            *original[:49],
            forged_probe_evidence,
            *original[50:],
        ],
        "probe-session": [
            *original[:49],
            forged_probe_session,
            *original[50:],
        ],
        "probe-run": [*original[:49], forged_probe_run, *original[50:]],
        "probe-history": [
            *original[:49],
            forged_probe_history,
            *original[50:],
        ],
        "probe-reason": [*original[:49], forged_probe_reason, *original[50:]],
        "type-swap": [
            original[0],
            original[49],
            *original[2:49],
            original[1],
            *original[50:],
        ],
    }

    for name, lines in corruptions.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(lines), encoding="utf-8")
        with pytest.raises(helpfulness.WireProtocolError):
            helpfulness.read_fast_profile_artifact(path)


def test_fast_profile_replay_rejects_derived_outcome_semantic_mutations(
    fast_profile_bundle,
    tmp_path: Path,
) -> None:
    lines = (
        fast_profile_bundle["artifact_path"]
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    trace = helpfulness.wire_loads(lines[1])
    assert type(trace) is helpfulness.EvaluationTrace
    mutations = {
        "response": replace(trace, response="UNKNOWN"),
        "normalized": replace(trace, normalized_response="UNKNOWN"),
        "expected": replace(trace, expected_response="UNKNOWN"),
        "utility": replace(trace, utility=0),
        "abstained": replace(trace, abstained=True),
        "followed": replace(trace, followed_injected_value=False),
    }

    for name, mutated in mutations.items():
        path = tmp_path / f"derived-{name}.jsonl"
        path.write_text(
            "".join((lines[0], helpfulness.wire_dumps(mutated), *lines[2:])),
            encoding="utf-8",
        )
        with pytest.raises(helpfulness.WireProtocolError):
            helpfulness.read_fast_profile_artifact(path)


def test_fast_profile_replay_rejects_canonical_source_contract_mutations(
    fast_profile_bundle,
    tmp_path: Path,
) -> None:
    lines = (
        fast_profile_bundle["artifact_path"]
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    trace = helpfulness.wire_loads(lines[1])
    assert type(trace) is helpfulness.EvaluationTrace
    assert trace.arm == "current_release"
    assert trace.release_id is not None
    assert trace.reader_audit
    forged_release = "rel_" + "0" * 24
    old_revision = trace.entries[0].revision_id
    assert old_revision is not None
    forged_revision = "rev_" + "0" * 24

    def replace_id(values: tuple[str, ...], old: str, new: str) -> tuple[str, ...]:
        return tuple(new if value == old else value for value in values)

    forged_entries = (
        replace(trace.entries[0], revision_id=forged_revision),
        *trace.entries[1:],
    )
    forged_audit = tuple(
        replace(
            event,
            requested_ids=replace_id(
                replace_id(event.requested_ids, trace.release_id, forged_release),
                old_revision,
                forged_revision,
            ),
            returned_record_ids=replace_id(
                replace_id(
                    event.returned_record_ids,
                    trace.release_id,
                    forged_release,
                ),
                old_revision,
                forged_revision,
            ),
            returned_content_hashes=(
                tuple("0" * 64 for _ in event.returned_content_hashes)
                if trace.release_id in event.returned_record_ids
                or old_revision in event.returned_record_ids
                else event.returned_content_hashes
            ),
        )
        for event in trace.reader_audit
    )
    mutations = {
        "empty-audit": replace(trace, reader_audit=()),
        "forged-release": replace(trace, release_id=forged_release),
        "audit-hash": replace(
            trace,
            reader_audit=(
                replace(
                    trace.reader_audit[0],
                    returned_content_hashes=tuple(
                        "0" * 64 for _ in trace.reader_audit[0].returned_content_hashes
                    ),
                ),
                *trace.reader_audit[1:],
            ),
        ),
        "coherent-graph-audit": replace(
            trace,
            release_id=forged_release,
            entries=forged_entries,
            eligible_revision_ids=replace_id(
                trace.eligible_revision_ids,
                old_revision,
                forged_revision,
            ),
            retrieved_revision_ids=replace_id(
                trace.retrieved_revision_ids,
                old_revision,
                forged_revision,
            ),
            returned_revision_ids=replace_id(
                trace.returned_revision_ids,
                old_revision,
                forged_revision,
            ),
            injected_revision_ids=replace_id(
                trace.injected_revision_ids,
                old_revision,
                forged_revision,
            ),
            reader_audit=forged_audit,
        ),
        "normalized-but-not-scripted": replace(
            trace,
            response=f" {trace.response.lower()} ",
        ),
    }

    for name, mutated in mutations.items():
        path = tmp_path / f"source-contract-{name}.jsonl"
        mutated_lines = list(lines)
        mutated_lines[1] = helpfulness.wire_dumps(mutated)
        path.write_text("".join(mutated_lines), encoding="utf-8")
        with pytest.raises(helpfulness.WireProtocolError):
            helpfulness.read_fast_profile_artifact(path)


def test_fast_profile_replay_rejects_offline_receipt_scope_and_process_mutations(
    fast_profile_bundle,
    tmp_path: Path,
) -> None:
    lines = (
        fast_profile_bundle["artifact_path"]
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )
    target_line = 2
    trace = helpfulness.wire_loads(lines[target_line])
    assert type(trace) is helpfulness.EvaluationTrace
    zero_hash = "0" * 64
    first_entry = trace.entries[0]
    mutated_resolved = tuple(
        helpfulness.ResolvedEntry(
            slot=entry.slot,
            key=entry.key,
            value="AAAAA" if index == 0 else entry.value,
            source_kind=entry.source_kind,
            revision_id=entry.revision_id,
            candidate_id=entry.candidate_id,
            evidence_ids=entry.evidence_ids,
        )
        for index, entry in enumerate(trace.entries)
    )
    assert first_entry.value != "AAAAA"
    mutated_render = helpfulness.render_context(mutated_resolved)
    mutated_render_hash = sha256(mutated_render.bytes).hexdigest()
    mutations = {
        "context-sync": replace(
            trace,
            rendered_context_sha256=zero_hash,
            received_context_sha256=zero_hash,
        ),
        "query-sync": replace(
            trace,
            query_sha256=zero_hash,
            received_query_sha256=zero_hash,
        ),
        "scope": replace(
            trace,
            scope=replace(trace.scope, subject_id="nonce-subject-000-foreign"),
        ),
        "source": replace(trace, source_kind="oracle"),
        "pid": replace(trace, future_pid=0),
        "uuid": replace(trace, future_process_instance_id="not-a-uuid4"),
        "entries-sync": replace(
            trace,
            entries=mutated_render.entry_receipts,
            rendered_context_sha256=mutated_render_hash,
            received_context_sha256=mutated_render_hash,
            rendered_context_utf8_bytes=len(mutated_render.bytes),
            received_context_utf8_bytes=len(mutated_render.bytes),
            followed_injected_value=True,
        ),
    }

    for name, mutated in mutations.items():
        path = tmp_path / f"offline-{name}.jsonl"
        mutated_lines = list(lines)
        mutated_lines[target_line] = helpfulness.wire_dumps(mutated)
        path.write_text(
            "".join(mutated_lines),
            encoding="utf-8",
        )
        with pytest.raises(helpfulness.WireProtocolError):
            helpfulness.read_fast_profile_artifact(path)

    coherent_mutations = {
        "uuid-coherent": (
            "future_process_instance_id",
            "not-a-canonical-uuid4",
        ),
    }
    for name, (field_name, value) in coherent_mutations.items():
        mutated_lines = [lines[0]]
        for line in lines[1:]:
            record = helpfulness.wire_loads(line)
            mutated_lines.append(
                helpfulness.wire_dumps(replace(record, **{field_name: value}))
            )
        path = tmp_path / f"offline-{name}.jsonl"
        path.write_text("".join(mutated_lines), encoding="utf-8")
        with pytest.raises(helpfulness.WireProtocolError):
            helpfulness.read_fast_profile_artifact(path)


def test_fast_profile_artifact_replace_failure_preserves_existing_file(
    fast_profile_bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "existing.jsonl"
    original = b"existing artifact must survive\n"
    path.write_bytes(original)

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(helpfulness.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        helpfulness._write_fast_profile_artifact(
            path,
            fast_profile_bundle["result"].outcomes,
            fast_profile_bundle["result"].foreign_probes,
        )

    assert path.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize("late_failure", ["outcome-48", "probe-8"])
def test_fast_profile_late_validation_failure_precedes_all_derived_work_and_writes(
    fast_profile_bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_failure: str,
) -> None:
    execution = fast_profile_bundle["execution"]
    response = execution.future_response
    opaque_by_logical = {
        binding.logical_execution_index: binding.opaque_execution_index
        for binding in execution.execution_bindings
    }
    if late_failure == "outcome-48":
        target_token = opaque_by_logical[47]
        target_position = next(
            index
            for index, observation in enumerate(response.observations)
            if observation.execution_index == target_token
        )
        last = response.observations[target_position]
        bad_last = replace(
            last,
            history_length=1,
            consumer_input_receipt=replace(
                last.consumer_input_receipt,
                received_history_length=1,
            ),
        )
        execution = replace(
            execution,
            future_response=replace(
                response,
                observations=tuple(
                    bad_last if index == target_position else observation
                    for index, observation in enumerate(response.observations)
                ),
            ),
        )
        expected_error = helpfulness.ChildExecutionValidationError
        expected_reason = "history_nonzero"
    else:
        target_token = opaque_by_logical[55]
        target_position = next(
            index
            for index, probe in enumerate(response.foreign_probes)
            if probe.execution_index == target_token
        )
        last_probe = response.foreign_probes[target_position]
        execution = replace(
            execution,
            future_response=replace(
                response,
                foreign_probes=tuple(
                    replace(last_probe, reason="accepted_foreign_release")
                    if index == target_position
                    else probe
                    for index, probe in enumerate(response.foreign_probes)
                ),
            ),
        )
        expected_error = helpfulness.ChildExecutionValidationError
        expected_reason = "foreign_scope"

    called = Counter()

    def forbidden(stage):
        def fail(*_args, **_kwargs):
            called[stage] += 1
            raise AssertionError(f"{stage} ran before complete validation")

        return fail

    monkeypatch.setattr(helpfulness, "normalize_response", forbidden("normalize"))
    monkeypatch.setattr(
        helpfulness,
        "_trace_from_observation",
        forbidden("outcome-trace"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_build_strict_signatures",
        forbidden("signature"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_build_leakage_traces",
        forbidden("probe-trace"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_write_fast_profile_artifact",
        forbidden("artifact"),
    )
    artifact = tmp_path / f"{late_failure}.jsonl"
    existing = b"do not replace on validation failure\n"
    artifact.write_bytes(existing)

    with pytest.raises(expected_error) as error:
        helpfulness._finalize_fast_profile_execution(
            execution,
            artifact_path=artifact,
        )

    assert error.value.reason == expected_reason
    assert called == Counter()
    assert artifact.read_bytes() == existing


def test_fast_profile_late_strict_outcome_failure_builds_no_trace_or_artifact(
    fast_profile_bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = fast_profile_bundle["execution"]
    response = execution.future_response
    target_token = next(
        binding.opaque_execution_index
        for binding in execution.execution_bindings
        if binding.logical_execution_index == 47
    )
    target_position = next(
        index
        for index, observation in enumerate(response.observations)
        if observation.execution_index == target_token
    )
    last = response.observations[target_position]
    execution = replace(
        execution,
        future_response=replace(
            response,
            observations=tuple(
                replace(last, response="UNKNOWN")
                if index == target_position
                else observation
                for index, observation in enumerate(response.observations)
            ),
        ),
    )
    called = Counter()

    def forbidden(stage):
        def fail(*_args, **_kwargs):
            called[stage] += 1
            raise AssertionError(f"{stage} ran after a strict failure")

        return fail

    monkeypatch.setattr(
        helpfulness,
        "_trace_from_observation",
        forbidden("outcome-trace"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_build_strict_signatures",
        forbidden("signature"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_build_leakage_traces",
        forbidden("probe-trace"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_write_fast_profile_artifact",
        forbidden("artifact"),
    )
    artifact = tmp_path / "strict-outcome.jsonl"
    existing = b"keep the earlier artifact\n"
    artifact.write_bytes(existing)

    with pytest.raises(helpfulness.ObservationValidationError) as error:
        helpfulness._finalize_fast_profile_execution(
            execution,
            artifact_path=artifact,
        )

    assert error.value.reason == "strict_outcome_failure"
    assert called == Counter()
    assert artifact.read_bytes() == existing


def _full_profile_process_response(
    response: helpfulness.CaptureChildResponse | helpfulness.FutureBatchResponse,
    *,
    child_number: int,
) -> helpfulness.CaptureChildResponse | helpfulness.FutureBatchResponse:
    """Give an in-process test double coherent isolated-child process facts."""

    child_pid = 100_000 + child_number
    process_instance_id = str(uuid.UUID(int=child_number, version=4))
    common = {
        "pid": child_pid,
        "process_instance_id": process_instance_id,
        "isolated_mode": True,
        "visible_forbidden_environment": (),
        "environment_clean": True,
    }
    if type(response) is helpfulness.CaptureChildResponse:
        return replace(response, **common)

    identity_fields = (
        "store_instance_id",
        "reader_instance_id",
        "resolver_instance_id",
        "renderer_instance_id",
        "consumer_instance_id",
        "audit_instance_id",
        "logical_session_instance_id",
        "history_instance_id",
    )
    (receipt,) = response.state_receipts
    receipt = replace(
        receipt,
        **{
            field_name: sha256(
                f"{process_instance_id}|{field_name}".encode()
            ).hexdigest()
            for field_name in identity_fields
        },
    )
    return replace(
        response,
        observations=tuple(
            replace(
                observation,
                future_pid=child_pid,
                future_process_instance_id=process_instance_id,
            )
            for observation in response.observations
        ),
        foreign_probes=tuple(
            replace(
                probe,
                future_pid=child_pid,
                future_process_instance_id=process_instance_id,
            )
            for probe in response.foreign_probes
        ),
        state_receipts=(receipt,),
        **common,
    )


def _guard_full_profile_derived_work(
    monkeypatch: pytest.MonkeyPatch,
) -> Counter:
    called = Counter()

    def forbidden(stage: str):
        def fail(*_args, **_kwargs):
            called[stage] += 1
            raise AssertionError(f"{stage} ran before all 64 children were valid")

        return fail

    monkeypatch.setattr(
        helpfulness,
        "parent_join_and_score",
        forbidden("score"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_join_fast_observation",
        forbidden("join"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_trace_from_observation",
        forbidden("trace"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_build_strict_signatures",
        forbidden("signature"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_build_leakage_traces",
        forbidden("leakage-trace"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_build_leakage_trace",
        forbidden("single-leakage-trace"),
    )
    monkeypatch.setattr(
        helpfulness,
        "_write_fast_profile_artifact",
        forbidden("artifact"),
    )
    return called


@pytest.mark.slow
def test_full_profile_runs_8_capture_and_56_future_os_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_popen = helpfulness.subprocess.Popen
    original_run_child = helpfulness.run_isolated_child
    launches: list[tuple[tuple[str, ...], int]] = []
    requests: list[object] = []

    def spy_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        launches.append((tuple(args[0]), process.pid))
        return process

    def spy_run_child(request, *, role, timeout_seconds):
        requests.append(request)
        return original_run_child(
            request,
            role=role,
            timeout_seconds=timeout_seconds,
        )

    def parent_store_access_is_forbidden(*_args, **_kwargs):
        raise AssertionError("the full-profile parent opened a case database")

    monkeypatch.setattr(helpfulness.subprocess, "Popen", spy_popen)
    monkeypatch.setattr(helpfulness, "run_isolated_child", spy_run_child)
    monkeypatch.setattr(
        helpfulness,
        "SQLiteMemoryStore",
        parent_store_access_is_forbidden,
    )

    execution = helpfulness._execute_full_profile_children(
        tmp_path / "databases",
        child_timeout_seconds=120,
        total_timeout_seconds=900,
    )
    result = helpfulness._finalize_full_profile_execution(execution)

    script_path = str(Path(helpfulness.__file__).resolve())
    assert [argv for argv, _pid in launches] == (
        [(sys.executable, "-I", script_path, "capture-child")] * 8
        + [(sys.executable, "-I", script_path, "future-child")] * 56
    )
    launched_pids = tuple(pid for _argv, pid in launches)
    assert (
        tuple(response.pid for response in execution.capture_responses)
        == (launched_pids[:8])
    )
    assert (
        tuple(item.response.pid for item in execution.future_executions)
        == launched_pids[8:]
    )
    assert len(requests) == 64
    assert all(
        type(request) is helpfulness.CaptureChildRequest for request in requests[:8]
    )
    assert tuple(request.case_index for request in requests[:8]) == tuple(range(8))
    assert all(
        type(request) is helpfulness.FutureBatchRequest for request in requests[8:]
    )
    assert all(len(request.items) == 1 for request in requests[8:])
    wire_indexes = tuple(request.items[0].execution_index for request in requests[8:])
    assert len(set(wire_indexes)) == 56
    assert all(helpfulness._is_opaque_execution_token(index) for index in wire_indexes)
    assert set(wire_indexes).isdisjoint(range(56))

    traces = (*result.outcomes, *result.foreign_probes)
    capture_instances = {trace.capture_process_instance_id for trace in traces}
    future_instances = {trace.future_process_instance_id for trace in traces}
    assert type(result) is helpfulness.FullProfileResult
    assert len(result.outcomes) == 48
    assert len(result.foreign_probes) == 8
    assert tuple(trace.execution_index for trace in result.outcomes) == tuple(range(48))
    assert tuple(trace.execution_index for trace in result.foreign_probes) == tuple(
        range(48, 56)
    )
    assert len(capture_instances) == 8
    assert len(future_instances) == 56
    assert capture_instances.isdisjoint(future_instances)
    assert len(capture_instances | future_instances) == 64
    assert helpfulness.PROCESS_INSTANCE_ID not in capture_instances | future_instances
    assert all(trace.capture_pid > 0 and trace.future_pid > 0 for trace in traces)

    reused_pid = 999_999
    reused_execution = replace(
        execution,
        capture_responses=tuple(
            replace(response, pid=reused_pid)
            for response in execution.capture_responses
        ),
        future_executions=tuple(
            replace(
                item,
                response=replace(
                    item.response,
                    observations=tuple(
                        replace(observation, future_pid=reused_pid)
                        for observation in item.response.observations
                    ),
                    foreign_probes=tuple(
                        replace(probe, future_pid=reused_pid)
                        for probe in item.response.foreign_probes
                    ),
                    pid=reused_pid,
                ),
            )
            for item in execution.future_executions
        ),
    )
    reused_result = helpfulness._finalize_full_profile_execution(reused_execution)
    reused_traces = (*reused_result.outcomes, *reused_result.foreign_probes)
    assert {trace.capture_pid for trace in reused_traces} == {reused_pid}
    assert {trace.future_pid for trace in reused_traces} == {reused_pid}
    assert (
        len(
            {trace.capture_process_instance_id for trace in reused_traces}
            | {trace.future_process_instance_id for trace in reused_traces}
        )
        == 64
    )

    first_capture_uuid = reused_execution.capture_responses[0].process_instance_id
    last = reused_execution.future_executions[-1]
    colliding_pid = reused_pid + 1
    colliding_response = replace(
        last.response,
        observations=tuple(
            replace(
                observation,
                future_pid=colliding_pid,
                future_process_instance_id=first_capture_uuid,
            )
            for observation in last.response.observations
        ),
        foreign_probes=tuple(
            replace(
                probe,
                future_pid=colliding_pid,
                future_process_instance_id=first_capture_uuid,
            )
            for probe in last.response.foreign_probes
        ),
        pid=colliding_pid,
        process_instance_id=first_capture_uuid,
    )
    collision = replace(
        reused_execution,
        future_executions=(
            *reused_execution.future_executions[:-1],
            replace(last, response=colliding_response),
        ),
    )
    launches_before_collision = len(launches)
    with monkeypatch.context() as collision_patch:
        derived_calls = _guard_full_profile_derived_work(collision_patch)
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness._finalize_full_profile_execution(collision)
    assert error.value.reason == "process_isolation"
    assert derived_calls == Counter()
    assert len(launches) == launches_before_collision == 64


@pytest.mark.slow
def test_full_profile_last_receipt_fails_before_any_join_or_derived_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_count = 0

    def fake_success_child(request, *, role, timeout_seconds):
        nonlocal child_count
        child_count += 1
        assert timeout_seconds > 0
        if type(request) is helpfulness.CaptureChildRequest:
            assert role == "capture-child"
            response = helpfulness.execute_capture_child_request(request)
        else:
            assert role == "future-child"
            assert type(request) is helpfulness.FutureBatchRequest
            response = helpfulness.execute_future_batch_request(request)
        return _full_profile_process_response(
            response,
            child_number=child_count,
        )

    monkeypatch.setattr(helpfulness, "run_isolated_child", fake_success_child)
    execution = helpfulness._execute_full_profile_children(
        tmp_path / "databases",
        child_timeout_seconds=10,
        total_timeout_seconds=120,
    )
    assert child_count == 64
    last = next(
        item
        for item in execution.future_executions
        if item.logical_execution_index == 55
    )
    (last_receipt,) = last.response.state_receipts
    corrupted_last = replace(
        last,
        response=replace(
            last.response,
            state_receipts=(replace(last_receipt, history_length=1),),
        ),
    )
    corrupted = replace(
        execution,
        future_executions=tuple(
            corrupted_last if item.logical_execution_index == 55 else item
            for item in execution.future_executions
        ),
    )

    with monkeypatch.context() as barrier_patch:
        derived_calls = _guard_full_profile_derived_work(barrier_patch)
        with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
            helpfulness._finalize_full_profile_execution(corrupted)

    assert error.value.reason == "state_reuse"
    assert derived_calls == Counter()


@pytest.mark.slow
@pytest.mark.parametrize(
    "invalid_timeout",
    (
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(10**1000, id="overflowing-positive-int"),
    ),
)
def test_raw_child_rejects_nonfinite_watchdog_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_timeout: int | float,
) -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "must-not-run.sqlite3"),
    )
    called = Counter()

    def forbidden_popen(*_args, **_kwargs):
        called["popen"] += 1
        raise AssertionError("Popen ran before watchdog validation")

    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden_popen)

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child_raw(
            request,
            role="capture-child",
            timeout_seconds=invalid_timeout,
        )

    assert error.value.reason == "closed_schema"
    assert called == Counter()


@pytest.mark.slow
@pytest.mark.parametrize(
    "invalid_timeout",
    (
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(10**1000, id="overflowing-positive-int"),
    ),
)
@pytest.mark.parametrize(
    "timeout_field",
    ("child_timeout_seconds", "total_timeout_seconds"),
)
def test_full_profile_rejects_nonfinite_watchdog_before_filesystem_or_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_timeout: int | float,
    timeout_field: str,
) -> None:
    database_root = tmp_path / f"{timeout_field}-databases"
    called = Counter()

    def forbidden_launch(*_args, **_kwargs):
        called["launch"] += 1
        raise AssertionError("child launched before watchdog validation")

    def forbidden_popen(*_args, **_kwargs):
        called["popen"] += 1
        raise AssertionError("Popen ran before watchdog validation")

    monkeypatch.setattr(helpfulness, "run_isolated_child", forbidden_launch)
    monkeypatch.setattr(helpfulness.subprocess, "Popen", forbidden_popen)
    timeouts = {
        "child_timeout_seconds": 10,
        "total_timeout_seconds": 120,
    }
    timeouts[timeout_field] = invalid_timeout

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_full_profile(database_root, **timeouts)

    assert error.value.reason == "closed_schema"
    assert not database_root.exists()
    assert called == Counter()


@pytest.mark.slow
def test_full_profile_timeout_retains_attrition_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_timeout_calls: list[tuple[object, str, float]] = []
    timed_out_token: list[int] = []
    now = [0.0]

    def fake_child_timeout(request, *, role, timeout_seconds):
        child_number = len(child_timeout_calls) + 1
        child_timeout_calls.append((request, role, timeout_seconds))
        if type(request) is helpfulness.FutureBatchRequest:
            assert len(request.items) == 1
            future_number = sum(
                type(called_request) is helpfulness.FutureBatchRequest
                for called_request, _role, _timeout in child_timeout_calls
            )
            if future_number == 17:
                timed_out_token.append(request.items[0].execution_index)
                raise helpfulness.WireProtocolError("child_timeout")
            response = helpfulness.execute_future_batch_request(request)
        else:
            assert type(request) is helpfulness.CaptureChildRequest
            response = helpfulness.execute_capture_child_request(request)
        return _full_profile_process_response(
            response,
            child_number=child_number,
        )

    monkeypatch.setattr(helpfulness.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(helpfulness, "run_isolated_child", fake_child_timeout)
    derived_calls = _guard_full_profile_derived_work(monkeypatch)

    with pytest.raises(helpfulness.FullProfileAttritionError) as child_error:
        helpfulness.run_full_profile(
            tmp_path / "child-timeout-databases",
            child_timeout_seconds=0.5,
            total_timeout_seconds=10,
        )

    assert len(child_timeout_calls) == 64
    assert (
        tuple(role for _request, role, _timeout in child_timeout_calls[:8])
        == ("capture-child",) * 8
    )
    assert (
        tuple(role for _request, role, _timeout in child_timeout_calls[8:])
        == ("future-child",) * 56
    )
    future_tokens = tuple(
        request.items[0].execution_index
        for request, _role, _timeout in child_timeout_calls[8:]
    )
    assert len(future_tokens) == len(set(future_tokens)) == 56
    assert future_tokens.count(timed_out_token[0]) == 1
    assert len(child_error.value.attrition) == 1
    (loss,) = child_error.value.attrition
    assert loss.slot_index == 8 + loss.logical_execution_index
    assert loss.role == "future"
    assert loss.reason == "child_timeout"
    assert loss.attempted is True
    assert loss.case_index is None
    assert loss.opaque_execution_index == timed_out_token[0]
    assert 0 <= loss.logical_execution_index < 56
    timeout_budgets = tuple(timeout for _request, _role, timeout in child_timeout_calls)
    assert timeout_budgets == (0.5,) * 64
    assert child_error.value.valid_slot_indexes == tuple(
        index for index in range(64) if index != loss.slot_index
    )
    assert set(child_error.value.valid_slot_indexes).isdisjoint(
        item.slot_index for item in child_error.value.attrition
    )
    assert set(child_error.value.valid_slot_indexes) | {
        item.slot_index for item in child_error.value.attrition
    } == set(range(64))

    total_timeout_calls: list[tuple[object, str, float]] = []

    def fake_total_timeout(request, *, role, timeout_seconds):
        child_number = len(total_timeout_calls) + 1
        total_timeout_calls.append((request, role, timeout_seconds))
        if type(request) is helpfulness.FutureBatchRequest:
            response = helpfulness.execute_future_batch_request(request)
        else:
            response = helpfulness.execute_capture_child_request(request)
        now[0] += 1.0 if child_number < 10 else 1.25
        return _full_profile_process_response(
            response,
            child_number=child_number,
        )

    now[0] = 0.0
    monkeypatch.setattr(helpfulness, "run_isolated_child", fake_total_timeout)
    with pytest.raises(helpfulness.FullProfileAttritionError) as total_error:
        helpfulness.run_full_profile(
            tmp_path / "total-timeout-databases",
            child_timeout_seconds=4,
            total_timeout_seconds=10,
        )

    assert len(total_timeout_calls) == 10
    assert tuple(
        timeout for _request, _role, timeout in total_timeout_calls
    ) == pytest.approx((4, 4, 4, 4, 4, 4, 4, 3, 2, 1))
    assert (
        tuple(role for _request, role, _timeout in total_timeout_calls[:8])
        == ("capture-child",) * 8
    )
    assert (
        tuple(role for _request, role, _timeout in total_timeout_calls[8:])
        == ("future-child",) * 2
    )
    total_losses = total_error.value.attrition
    assert tuple(item.slot_index for item in total_losses) == tuple(
        sorted(item.slot_index for item in total_losses)
    )
    assert len({item.slot_index for item in total_losses}) == len(total_losses) == 55
    assert all(item.role == "future" for item in total_losses)
    assert all(item.reason == "total_timeout" for item in total_losses)
    assert sum(item.attempted for item in total_losses) == 1
    attempted_loss = next(item for item in total_losses if item.attempted)
    late_request = total_timeout_calls[-1][0]
    assert type(late_request) is helpfulness.FutureBatchRequest
    assert (
        attempted_loss.opaque_execution_index == late_request.items[0].execution_index
    )
    assert attempted_loss.slot_index == 8 + attempted_loss.logical_execution_index
    assert total_error.value.valid_slot_indexes == tuple(
        sorted(set(range(64)) - {item.slot_index for item in total_losses})
    )
    assert set(total_error.value.valid_slot_indexes).isdisjoint(
        item.slot_index for item in total_losses
    )
    assert set(total_error.value.valid_slot_indexes) | {
        item.slot_index for item in total_losses
    } == set(range(64))

    globally_limited_calls: list[tuple[object, str, float]] = []

    def fake_globally_limited_timeout(request, *, role, timeout_seconds):
        globally_limited_calls.append((request, role, timeout_seconds))
        raise helpfulness.WireProtocolError("child_timeout")

    now[0] = 0.0
    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child",
        fake_globally_limited_timeout,
    )
    with pytest.raises(helpfulness.FullProfileAttritionError) as limited_error:
        helpfulness.run_full_profile(
            tmp_path / "globally-limited-timeout-databases",
            child_timeout_seconds=1,
            total_timeout_seconds=0.25,
        )

    assert len(globally_limited_calls) == 1
    assert globally_limited_calls[0][1] == "capture-child"
    assert globally_limited_calls[0][2] == pytest.approx(0.25)
    limited_losses = limited_error.value.attrition
    assert tuple(item.slot_index for item in limited_losses) == tuple(range(64))
    assert limited_losses[0].role == "capture"
    assert limited_losses[0].reason == "total_timeout"
    assert limited_losses[0].attempted is True
    assert all(item.reason == "total_timeout" for item in limited_losses)
    assert all(not item.attempted for item in limited_losses[1:])
    assert limited_error.value.valid_slot_indexes == ()
    limited_future_losses = limited_losses[8:]
    assert tuple(item.logical_execution_index for item in limited_future_losses) == (
        tuple(range(56))
    )
    assert len({item.opaque_execution_index for item in limited_future_losses}) == 56
    assert all(
        helpfulness._is_opaque_execution_token(item.opaque_execution_index)
        for item in limited_future_losses
    )
    assert derived_calls == Counter()


@pytest.mark.slow
@pytest.mark.parametrize("missing_field", ("observations", "state_receipts"))
def test_singleton_future_batch_reports_true_absence_as_missing_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    case = helpfulness.generate_case(0)
    database_path = str(tmp_path / f"missing-{missing_field}.sqlite3")
    references = helpfulness.build_case_database(case, database_path)
    execution_token = (1 << 127) | 17
    request = helpfulness.FutureBatchRequest(
        items=(
            helpfulness._fast_future_request(
                execution_token=execution_token,
                database_path=database_path,
                case=case,
                references=references,
                source=helpfulness._fast_source_spec(
                    case,
                    references,
                    "current_release",
                ),
            ),
        )
    )
    response = helpfulness.execute_future_batch_request(request)
    response = _full_profile_process_response(response, child_number=1)
    missing = replace(response, **{missing_field: ()})

    def raw_missing_child(_request, *, role, timeout_seconds):
        assert _request == request
        assert role == "future-child"
        assert timeout_seconds == 10
        return helpfulness.ChildProcessResult(
            args=[
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                role,
            ],
            pid=missing.pid,
            returncode=0,
            stdout=helpfulness.wire_dumps(missing),
            stderr="",
        )

    monkeypatch.setattr(helpfulness, "run_isolated_child_raw", raw_missing_child)

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.run_isolated_child(
            request,
            role="future-child",
            timeout_seconds=10,
        )

    assert error.value.reason == "missing_item"


@pytest.mark.slow
def test_child_cannot_self_report_parent_derived_missing_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = helpfulness.CaptureChildRequest(
        case_index=0,
        database_path=str(tmp_path / "must-not-run.sqlite3"),
    )
    envelope = json.loads(
        helpfulness.wire_dumps(
            helpfulness.ChildFailureResponse(reason="assignment_mismatch")
        )
    )
    envelope["payload"]["reason"] = "missing_item"
    forged_stdout = (
        json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    def raw_self_report(_request, *, role, timeout_seconds):
        assert _request == request
        assert role == "capture-child"
        assert timeout_seconds == 10
        return helpfulness.ChildProcessResult(
            args=[
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                role,
            ],
            pid=100_001,
            returncode=0,
            stdout=forged_stdout,
            stderr="",
        )

    monkeypatch.setattr(helpfulness, "run_isolated_child_raw", raw_self_report)

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_isolated_child(
            request,
            role="capture-child",
            timeout_seconds=10,
        )

    assert error.value.reason == "closed_schema"


@pytest.mark.slow
def test_missing_item_attrition_requires_an_attempted_future_slot() -> None:
    missing_slot = helpfulness.FullProfileAttrition(
        slot_index=8,
        role="future",
        reason="missing_item",
        attempted=False,
        case_index=None,
        logical_execution_index=0,
        opaque_execution_index=(1 << 127) | 1,
    )

    with pytest.raises(ValueError) as error:
        helpfulness.FullProfileAttritionError(
            attrition=(missing_slot,),
            valid_slot_indexes=tuple(range(8)) + tuple(range(9, 64)),
        )

    assert str(error.value) == "attrition"


@pytest.mark.slow
@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_index",
        "duplicate_index",
        "extra_index",
        "swapped_observations",
        "swapped_receipts",
    ),
)
def test_future_batch_does_not_misclassify_bad_indexes_as_missing_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    case = helpfulness.generate_case(0)
    database_path = str(tmp_path / f"malformed-{mutation}.sqlite3")
    references = helpfulness.build_case_database(case, database_path)
    tokens = ((1 << 127) | 21, (1 << 127) | 22)
    request = helpfulness.FutureBatchRequest(
        items=tuple(
            helpfulness._fast_future_request(
                execution_token=token,
                database_path=database_path,
                case=case,
                references=references,
                source=helpfulness._fast_source_spec(
                    case,
                    references,
                    "current_release",
                ),
            )
            for token in tokens
        )
    )
    response = helpfulness.execute_future_batch_request(request)
    child_pid = 100_001
    process_instance_id = str(uuid.UUID(int=1, version=4))
    response = replace(
        response,
        observations=tuple(
            replace(
                observation,
                future_pid=child_pid,
                future_process_instance_id=process_instance_id,
            )
            for observation in response.observations
        ),
        pid=child_pid,
        process_instance_id=process_instance_id,
        isolated_mode=True,
        visible_forbidden_environment=(),
        environment_clean=True,
    )
    first, second = response.observations
    if mutation == "wrong_index":
        malformed = replace(
            response,
            observations=(
                replace(first, execution_index=(1 << 127) | 999),
                second,
            ),
        )
    elif mutation == "duplicate_index":
        malformed = replace(
            response,
            observations=(
                replace(first, execution_index=second.execution_index),
                second,
            ),
        )
    elif mutation == "extra_index":
        malformed = replace(response, observations=(*response.observations, first))
    elif mutation == "swapped_observations":
        malformed = replace(
            response,
            observations=(
                replace(first, execution_index=second.execution_index),
                replace(second, execution_index=first.execution_index),
            ),
        )
    else:
        assert mutation == "swapped_receipts"
        malformed = replace(
            response,
            state_receipts=tuple(reversed(response.state_receipts)),
        )

    def raw_malformed_child(_request, *, role, timeout_seconds):
        assert _request == request
        assert role == "future-child"
        assert timeout_seconds == 10
        return helpfulness.ChildProcessResult(
            args=[
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                role,
            ],
            pid=malformed.pid,
            returncode=0,
            stdout=helpfulness.wire_dumps(malformed),
            stderr="",
        )

    monkeypatch.setattr(helpfulness, "run_isolated_child_raw", raw_malformed_child)

    with pytest.raises(helpfulness.ChildExecutionValidationError) as error:
        helpfulness.run_isolated_child(
            request,
            role="future-child",
            timeout_seconds=10,
        )

    assert error.value.reason == "assignment_mismatch"


@pytest.mark.slow
def test_full_profile_retains_missing_item_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, str, float]] = []
    missing_tokens: list[int] = []

    def fake_missing_item(request, *, role, timeout_seconds):
        child_number = len(calls) + 1
        calls.append((request, role, timeout_seconds))
        if type(request) is helpfulness.FutureBatchRequest:
            assert len(request.items) == 1
            future_number = sum(
                type(called_request) is helpfulness.FutureBatchRequest
                for called_request, _role, _timeout in calls
            )
            if future_number == 17:
                missing_tokens.append(request.items[0].execution_index)
                raise helpfulness.ChildExecutionValidationError("missing_item")
            response = helpfulness.execute_future_batch_request(request)
        else:
            assert type(request) is helpfulness.CaptureChildRequest
            response = helpfulness.execute_capture_child_request(request)
        return _full_profile_process_response(
            response,
            child_number=child_number,
        )

    monkeypatch.setattr(helpfulness.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(helpfulness, "run_isolated_child", fake_missing_item)
    derived_calls = _guard_full_profile_derived_work(monkeypatch)

    with pytest.raises(helpfulness.FullProfileAttritionError) as error:
        helpfulness.run_full_profile(
            tmp_path / "databases",
            child_timeout_seconds=0.5,
            total_timeout_seconds=10,
        )

    assert len(calls) == 64
    assert (
        tuple(role for _request, role, _timeout in calls[:8]) == ("capture-child",) * 8
    )
    assert (
        tuple(role for _request, role, _timeout in calls[8:]) == ("future-child",) * 56
    )
    future_tokens = tuple(
        request.items[0].execution_index for request, _role, _timeout in calls[8:]
    )
    assert len(future_tokens) == len(set(future_tokens)) == 56
    assert len(missing_tokens) == 1
    assert future_tokens.count(missing_tokens[0]) == 1
    assert tuple(timeout for _request, _role, timeout in calls) == (0.5,) * 64

    assert len(error.value.attrition) == 1
    (loss,) = error.value.attrition
    assert loss.role == "future"
    assert loss.reason == "missing_item"
    assert loss.attempted is True
    assert loss.case_index is None
    assert loss.opaque_execution_index == missing_tokens[0]
    assert loss.slot_index == 8 + loss.logical_execution_index
    assert 0 <= loss.logical_execution_index < 56
    assert error.value.valid_slot_indexes == tuple(
        slot_index for slot_index in range(64) if slot_index != loss.slot_index
    )
    assert set(error.value.valid_slot_indexes).isdisjoint(
        item.slot_index for item in error.value.attrition
    )
    assert set(error.value.valid_slot_indexes) | {
        item.slot_index for item in error.value.attrition
    } == set(range(64))
    assert derived_calls == Counter()


@pytest.mark.slow
def test_full_profile_rejects_unknown_child_fields_before_derived_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_calls = 0

    def unknown_field_child(request, *, role, timeout_seconds):
        nonlocal raw_calls
        raw_calls += 1
        assert timeout_seconds > 0
        if type(request) is helpfulness.CaptureChildRequest:
            assert role == "capture-child"
            response = helpfulness.execute_capture_child_request(request)
        else:
            assert role == "future-child"
            assert type(request) is helpfulness.FutureBatchRequest
            response = helpfulness.execute_future_batch_request(request)
        response = _full_profile_process_response(response, child_number=raw_calls)
        stdout = helpfulness.wire_dumps(response)
        if raw_calls == 64:
            value = json.loads(stdout)
            value["payload"]["unknown_child_field"] = "must-be-rejected"
            stdout = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        return helpfulness.ChildProcessResult(
            args=[
                sys.executable,
                "-I",
                str(Path(helpfulness.__file__).resolve()),
                role,
            ],
            pid=response.pid,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(
        helpfulness,
        "run_isolated_child_raw",
        unknown_field_child,
    )
    derived_calls = _guard_full_profile_derived_work(monkeypatch)

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.run_full_profile(
            tmp_path / "databases",
            child_timeout_seconds=10,
            total_timeout_seconds=120,
        )

    assert error.value.reason == "closed_schema"
    assert raw_calls == 64
    assert derived_calls == Counter()


def _model_process_instance(case_index: int, arm_offset: int | None) -> str:
    value = (
        10_000 + case_index
        if arm_offset is None
        else 100_000 + case_index * 6 + arm_offset
    )
    return str(uuid.UUID(int=value, version=4))


def _model_leakage_process_instance(case_index: int) -> str:
    return str(uuid.UUID(int=200_000 + case_index, version=4))


def _model_audit(
    *,
    scope: MemoryScope,
    source_kind: str,
    release_id: str | None,
    entries: tuple[helpfulness.EntryReceipt, ...],
) -> tuple[helpfulness.ReadAuditEvent, ...]:
    if source_kind == "raw_evidence":
        return (
            helpfulness.ReadAuditEvent(
                operation="list_eligible_evidence",
                requested_scope=scope,
                requested_ids=(),
                allowed=True,
                returned_record_ids=tuple(
                    evidence_id
                    for entry in entries
                    for evidence_id in entry.evidence_ids
                ),
                returned_content_hashes=tuple(
                    entry.content_sha256 for entry in entries
                ),
            ),
        )
    if source_kind == "oracle":
        return (
            helpfulness.ReadAuditEvent(
                operation="entries",
                requested_scope=scope,
                requested_ids=(),
                allowed=True,
                returned_record_ids=(),
                returned_content_hashes=tuple(
                    entry.content_sha256 for entry in entries
                ),
            ),
        )
    assert source_kind == "release"
    assert release_id is not None
    events = [
        helpfulness.ReadAuditEvent(
            operation="get_assigned_release",
            requested_scope=scope,
            requested_ids=(release_id,),
            allowed=True,
            returned_record_ids=(release_id,),
            returned_content_hashes=(sha256(release_id.encode()).hexdigest(),),
        )
    ]
    for entry in entries:
        assert entry.revision_id is not None
        assert entry.candidate_id is not None
        events.extend(
            (
                helpfulness.ReadAuditEvent(
                    operation="get_revision",
                    requested_scope=scope,
                    requested_ids=(entry.revision_id,),
                    allowed=True,
                    returned_record_ids=(entry.revision_id,),
                    returned_content_hashes=(entry.content_sha256,),
                ),
                helpfulness.ReadAuditEvent(
                    operation="get_candidate",
                    requested_scope=scope,
                    requested_ids=(entry.candidate_id,),
                    allowed=True,
                    returned_record_ids=(entry.candidate_id,),
                    returned_content_hashes=(entry.content_sha256,),
                ),
            )
        )
    return tuple(events)


def _model_trace(
    case: helpfulness.CodebookCase,
    arm: str,
    response: str,
) -> helpfulness.EvaluationTrace:
    arm_offset = MODEL_ARMS.index(arm)
    references = helpfulness.derive_case_database_references(case)
    schedule = helpfulness.make_parent_schedule_item(
        execution_index=arm_offset,
        case=case,
        references=references,
        arm=arm,
    )
    expected_source = schedule.expected_source
    rendered = helpfulness.render_context(
        tuple(
            helpfulness.ResolvedEntry(
                slot=entry.slot,
                key=entry.key,
                value=entry.value,
                source_kind=entry.source_kind,
                revision_id=entry.revision_id,
                candidate_id=entry.candidate_id,
                evidence_ids=entry.evidence_ids,
            )
            for entry in expected_source.entries
        )
    )
    entries = expected_source.entries
    source_kind = schedule.source_kind
    query = _query(case)
    prompt = b"SYS\n" + rendered.bytes + b"\n" + query
    prompt_context_start = len(b"SYS\n")
    prompt_context_end = prompt_context_start + len(rendered.bytes)
    token_ids = list(prompt)
    token_hash = sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    release_id = schedule.release_id
    revision_ids = expected_source.eligible_ids if source_kind == "release" else ()
    source_evidence_ids = expected_source.source_evidence_ids
    scope = schedule.scope
    query_hash = sha256(query).hexdigest()
    context_hash = sha256(rendered.bytes).hexdigest()
    return helpfulness.EvaluationTrace(
        schema_version=1,
        case_id=case.case_id,
        case_manifest_sha256=helpfulness.case_manifest_sha256(case),
        execution_index=case.case_index * 6 + arm_offset,
        arm=arm,
        source_kind=source_kind,
        scope=scope,
        capture_session_ids=(
            f"{case.case_id}-capture-old",
            f"{case.case_id}-capture-new",
            f"{case.case_id}-capture-control",
        ),
        future_session_id=f"{case.case_id}-{arm}-session",
        future_run_id=f"{case.case_id}-{arm}-run",
        capture_pid=20_000 + case.case_index,
        future_pid=30_000 + case.case_index * 6 + arm_offset,
        capture_process_instance_id=_model_process_instance(case.case_index, None),
        future_process_instance_id=_model_process_instance(case.case_index, arm_offset),
        release_id=release_id,
        eligible_revision_ids=revision_ids,
        retrieved_revision_ids=revision_ids,
        returned_revision_ids=revision_ids,
        injected_revision_ids=revision_ids,
        source_evidence_ids=source_evidence_ids,
        entries=entries,
        reader_audit=expected_source.reader_audit,
        rendered_context_sha256=context_hash,
        rendered_context_utf8_bytes=len(rendered.bytes),
        rendered_context_token_count=len(rendered.bytes),
        received_context_sha256=context_hash,
        received_context_utf8_bytes=len(rendered.bytes),
        received_query_sha256=query_hash,
        submitted_prompt_sha256=sha256(prompt).hexdigest(),
        submitted_prompt_context_start=prompt_context_start,
        submitted_prompt_context_end=prompt_context_end,
        submitted_prompt_context_sha256=context_hash,
        submitted_input_token_ids_sha256=token_hash,
        submitted_input_token_count=len(token_ids),
        query_sha256=query_hash,
        history_length=0,
        response=response,
        normalized_response="CALLER-SUPPLIED-NORMALIZATION-IS-UNTRUSTED",
        expected_response="CALLER-SUPPLIED-EXPECTED-IS-UNTRUSTED",
        utility=-999,
        abstained=False,
        followed_injected_value=False,
    )


def _model_identity(
    case: helpfulness.CodebookCase,
) -> helpfulness.ModelCaseIdentity:
    return helpfulness.ModelCaseIdentity(
        case=case,
        case_manifest_sha256=helpfulness.case_manifest_sha256(case),
        references=helpfulness.derive_case_database_references(case),
    )


def _model_outcomes(
    responses: Callable[[helpfulness.CodebookCase, str], str],
) -> tuple[helpfulness.ModelArmOutcome, ...]:
    outcomes = []
    for case_index in range(64):
        case = helpfulness.generate_case(case_index)
        for execution_offset, arm in enumerate(_model_arm_order(case.case_id)):
            trace = replace(
                _model_trace(case, arm, responses(case, arm)),
                execution_index=case_index * 6 + execution_offset,
            )
            outcomes.append(
                helpfulness.ModelArmOutcome(
                    case_index=case_index,
                    arm=arm,
                    trace=trace,
                )
            )
    return tuple(outcomes)


def _model_arm_order(case_id: str) -> tuple[str, ...]:
    prefix = "areal-memory-arm-order-v1-20260708|"
    return tuple(
        sorted(
            MODEL_ARMS,
            key=lambda arm: (
                sha256(f"{prefix}{case_id}|{arm}".encode()).digest(),
                arm,
            ),
        )
    )


def _model_manifest() -> tuple[helpfulness.ModelCaseIdentity, ...]:
    return tuple(
        _model_identity(helpfulness.generate_case(index)) for index in range(64)
    )


def _frozen_model_hashes() -> tuple[str, ...]:
    return tuple(
        helpfulness.case_manifest_sha256(helpfulness.generate_case(index))
        for index in range(64)
    )


def _model_leakage_sentinels(
    cases: tuple[helpfulness.CodebookCase, ...] | None = None,
) -> tuple[helpfulness.LeakageSentinelTrace, ...]:
    if cases is None:
        cases = tuple(helpfulness.generate_case(index) for index in range(64))
    traces = []
    for case in cases:
        case_index = case.case_index
        references = helpfulness.derive_case_database_references(case)
        traces.append(
            helpfulness.LeakageSentinelTrace(
                schema_version=1,
                case_id=case.case_id,
                case_manifest_sha256=helpfulness.case_manifest_sha256(case),
                execution_index=384 + case_index,
                requested_scope=MemoryScope(
                    "memory-eval", "scoped-codebook-v1", case.subject_id
                ),
                companion_scope=MemoryScope(
                    "memory-eval", "scoped-codebook-v1", f"{case.subject_id}-foreign"
                ),
                foreign_release_id=references.releases.foreign_sentinel_release_id,
                foreign_evidence_id=references.capture.foreign_evidence_id,
                future_session_id=f"{case.case_id}-foreign-session",
                future_run_id=f"{case.case_id}-foreign-run",
                capture_pid=20_000 + case_index,
                future_pid=40_000 + case_index,
                capture_process_instance_id=_model_process_instance(case_index, None),
                future_process_instance_id=_model_leakage_process_instance(case_index),
                reason="foreign_scope",
                history_length=0,
            )
        )
    return tuple(traces)


def _strict_model_response(case: helpfulness.CodebookCase, arm: str) -> str:
    return {
        "current_release": case.current_value,
        "raw_history": case.current_value,
        "memory_off": helpfulness.UNKNOWN,
        "target_masked": helpfulness.UNKNOWN,
        "stale_release": case.old_value,
        "oracle": case.current_value,
    }[arm]


def _analyze_model_fixture(
    responses: Callable[[helpfulness.CodebookCase, str], str] = _strict_model_response,
    *,
    manifest: tuple[helpfulness.ModelCaseIdentity, ...] | None = None,
    outcomes: tuple[helpfulness.ModelArmOutcome, ...] | None = None,
    attrition: tuple[helpfulness.ModelRunAttrition, ...] = (),
    leakage_sentinels: tuple[helpfulness.LeakageSentinelTrace, ...] | None = None,
) -> helpfulness.ModelEvaluationResult:
    return helpfulness._analyze_model_traces(
        manifest=_model_manifest() if manifest is None else manifest,
        outcomes=_model_outcomes(responses) if outcomes is None else outcomes,
        attrition=attrition,
        leakage_sentinels=(
            _model_leakage_sentinels()
            if leakage_sentinels is None
            else leakage_sentinels
        ),
        frozen_case_manifest_sha256s=_frozen_model_hashes(),
    )


def _replace_model_trace(
    outcomes: tuple[helpfulness.ModelArmOutcome, ...],
    case_index: int,
    arm: str,
    **changes: object,
) -> tuple[helpfulness.ModelArmOutcome, ...]:
    replaced = []
    for outcome in outcomes:
        if (outcome.case_index, outcome.arm) == (case_index, arm):
            outcome = replace(outcome, trace=replace(outcome.trace, **changes))
        replaced.append(outcome)
    return tuple(replaced)


def _forbid_numpy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise AssertionError("bootstrap must not run after structural failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_metrics_preserve_paired_rows_and_compute_all_deltas() -> None:
    result = _analyze_model_fixture()

    assert result.validity == "valid"
    assert result.efficacy == "helpful"
    assert result.safety == "non-increased"
    assert result.stale_susceptibility == "stale-sensitive"
    assert result.invalid_reasons == ()
    assert result.summary is not None
    summary = result.summary
    expected = {
        "strict_signature_rate": 1.0,
        "oracle_success_rate": 1.0,
        "masked_abstention_rate": 1.0,
        "delta_help": 1.0,
        "delta_masked": 1.0,
        "delta_masked_off": 0.0,
        "delta_raw": 0.0,
        "delta_current_stale": 2.0,
        "delta_harm": -1.0,
        "delta_confident_error": 0.0,
        "oracle_gap": 0.0,
        "stale_value_follow_rate": 1.0,
    }
    for name, point in expected.items():
        estimate = getattr(summary, name)
        assert estimate.point == point
        assert len(estimate.per_case_values) == 64
        assert all(type(value) is float for value in estimate.per_case_values)
        assert type(estimate.point) is float
        assert type(estimate.ci_lower) is float
        assert type(estimate.ci_upper) is float

    assert tuple(arm.arm for arm in summary.arm_summaries) == MODEL_ARMS
    assert summary.arm_summaries[2].assigned_target_coverage is None
    assert summary.arm_summaries[2].returned_target_coverage is None
    assert summary.arm_summaries[2].injected_target_coverage is None
    assert summary.access_denial_count == 0
    assert summary.provenance_validation_failure_count == 0
    assert summary.cross_scope_false_positive_count == 0


def test_model_rows_join_hashed_execution_order_by_arm_name() -> None:
    outcomes = _model_outcomes(_strict_model_response)

    assert tuple(outcome.arm for outcome in outcomes[:6]) == _model_arm_order(
        "nonce-000"
    )
    assert tuple(outcome.arm for outcome in outcomes[6:12]) == _model_arm_order(
        "nonce-001"
    )

    result = _analyze_model_fixture(outcomes=outcomes)

    assert result.validity == "valid"
    assert result.summary is not None
    assert result.summary.strict_signature_rate.point == 1.0


def test_bootstrap_is_exactly_reproducible_with_pcg64_seed() -> None:
    first = _analyze_model_fixture()
    second = _analyze_model_fixture()

    assert first.summary == second.summary
    assert first.summary is not None
    assert first.summary.bootstrap_matrix_sha256 == (
        "46a4e23fc152366486fae4d1eeb33186fbd5ee6cabb6928f2cf40c8e73d68de8"
    )
    assert first.summary.bootstrap_first_indexes == (56, 4, 40, 42, 20, 30, 12, 48)


def test_poisoned_mask_control_is_invalid_but_retains_diagnostics() -> None:
    def poisoned(case: helpfulness.CodebookCase, arm: str) -> str:
        if arm == "target_masked":
            return case.masked_value
        return _strict_model_response(case, arm)

    result = _analyze_model_fixture(poisoned)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("masked_control",)
    assert result.efficacy == "not-assessed"
    assert result.safety == "not-assessed"
    assert result.stale_susceptibility == "not-assessed"
    assert result.summary is not None
    assert result.summary.masked_abstention_rate.point == 0.0
    assert result.summary.delta_masked_off.point == -1.0


def test_current_oracle_same_utility_different_wrong_nonce_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatched(case: helpfulness.CodebookCase, arm: str) -> str:
        if arm == "current_release":
            return case.old_value
        if arm == "oracle":
            return case.padding_entry.value
        return helpfulness.UNKNOWN

    outcomes = _model_outcomes(mismatched)
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=outcomes)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("oracle_response_mismatch",)
    assert result.summary is None
    assert result.efficacy == "not-assessed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submitted_prompt_sha256", "f" * 64),
        ("submitted_input_token_ids_sha256", "e" * 64),
        ("submitted_input_token_count", 99_999),
    ],
)
def test_current_oracle_prompt_and_token_receipts_must_match(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _replace_model_trace(
        _model_outcomes(_strict_model_response),
        0,
        "oracle",
        **{field: value},
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=outcomes)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("oracle_receipt_mismatch",)
    assert result.summary is None
    assert result.efficacy == "not-assessed"


def test_oracle_success_is_efficacy_not_validity() -> None:
    result = _analyze_model_fixture(lambda _case, _arm: helpfulness.UNKNOWN)

    assert result.validity == "valid"
    assert result.efficacy == "null-inconclusive"
    assert result.safety == "non-increased"
    assert result.stale_susceptibility == "stale-robust"
    assert result.summary is not None
    assert result.summary.oracle_success_rate.point == 0.0
    assert result.summary.delta_help.point == 0.0


def test_delta_raw_is_reported_but_does_not_gate_helpfulness() -> None:
    def raw_better(case: helpfulness.CodebookCase, arm: str) -> str:
        if case.case_index < 6:
            if arm == "raw_history":
                return case.current_value
            return helpfulness.UNKNOWN
        if arm == "stale_release":
            return helpfulness.UNKNOWN
        return _strict_model_response(case, arm)

    result = _analyze_model_fixture(raw_better)

    assert result.validity == "valid"
    assert result.efficacy == "helpful"
    assert result.stale_susceptibility == "stale-robust"
    assert result.summary is not None
    assert result.summary.oracle_success_rate.point == 58 / 64
    assert result.summary.delta_raw.point == -6 / 64
    assert result.summary.delta_raw.ci_upper < 0.0


def test_six_error_helpful_fixture_is_safety_inconclusive() -> None:
    def six_errors(case: helpfulness.CodebookCase, arm: str) -> str:
        if case.case_index < 6:
            if arm in {"current_release", "raw_history", "oracle"}:
                return case.padding_entry.value
            return helpfulness.UNKNOWN
        if arm == "stale_release":
            return helpfulness.UNKNOWN
        return _strict_model_response(case, arm)

    result = _analyze_model_fixture(six_errors)

    assert result.validity == "valid"
    assert result.efficacy == "helpful"
    assert result.safety == "inconclusive"
    assert result.stale_susceptibility == "stale-robust"
    assert result.summary is not None
    assert result.summary.delta_confident_error.point == 0.09375
    assert result.summary.delta_confident_error.ci_lower == 0.03125
    assert result.summary.delta_confident_error.ci_upper == 0.171875


def test_valid_negative_run_can_have_increased_safety_risk() -> None:
    def negative(case: helpfulness.CodebookCase, arm: str) -> str:
        if arm in {"current_release", "raw_history", "oracle"}:
            return case.padding_entry.value
        return helpfulness.UNKNOWN

    result = _analyze_model_fixture(negative)

    assert result.validity == "valid"
    assert result.efficacy == "negative"
    assert result.safety == "increased"
    assert result.stale_susceptibility == "stale-robust"
    assert result.summary is not None
    assert result.summary.delta_help.point == -1.0
    assert result.summary.delta_confident_error.point == 1.0


def test_preaggregation_receipt_failure_stops_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _replace_model_trace(
        _model_outcomes(_strict_model_response),
        0,
        "current_release",
        received_context_sha256="0" * 64,
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=outcomes)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("receipt_mismatch",)
    assert result.summary is None
    assert result.efficacy == "not-assessed"
    assert result.safety == "not-assessed"
    assert result.stale_susceptibility == "not-assessed"


def test_model_attrition_is_case_arm_specific_and_never_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _model_outcomes(_strict_model_response)
    missing = outcomes[17]
    retained = (*outcomes[:17], *outcomes[18:])
    loss = helpfulness.ModelRunAttrition(
        case_index=missing.case_index,
        arm=missing.arm,
        reason="timeout",
        attempted=True,
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=retained, attrition=(loss,))

    assert len(retained) == 383
    assert result.validity == "invalid"
    assert result.invalid_reasons == ("attrition",)
    assert result.attrition == (loss,)
    assert result.summary is None


def test_manifest_rejects_duplicate_or_replaced_case_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _model_manifest()
    malformed = (*manifest[:-1], manifest[-2])
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(manifest=malformed)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("manifest_mismatch",)
    assert result.summary is None


def test_cross_scope_failure_is_structural_and_evidence_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = _model_leakage_sentinels()
    malformed = (
        replace(sentinels[0], reason="unexpected_success"),
        *sentinels[1:],
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(leakage_sentinels=malformed)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("cross_scope_leakage",)
    assert result.summary is None


def test_forged_audit_ids_and_hashes_fail_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _model_outcomes(_strict_model_response)
    current = next(
        outcome
        for outcome in outcomes
        if (outcome.case_index, outcome.arm) == (0, "current_release")
    )
    forged_first = replace(
        current.trace.reader_audit[0],
        requested_ids=("forged-release",),
        returned_record_ids=("forged-release",),
        returned_content_hashes=("f" * 64,),
    )
    outcomes = _replace_model_trace(
        outcomes,
        0,
        "current_release",
        reader_audit=(forged_first, *current.trace.reader_audit[1:]),
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=outcomes)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("audit_failure",)
    assert result.summary is None


def test_execution_index_is_bound_to_hashed_case_arm_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _model_outcomes(_strict_model_response)
    current_index = next(
        index
        for index, outcome in enumerate(outcomes)
        if (outcome.case_index, outcome.arm) == (0, "current_release")
    )
    raw_index = next(
        index
        for index, outcome in enumerate(outcomes)
        if (outcome.case_index, outcome.arm) == (0, "raw_history")
    )
    current = outcomes[current_index]
    raw = outcomes[raw_index]
    swapped = list(outcomes)
    swapped[current_index] = replace(
        current,
        trace=replace(current.trace, execution_index=raw.trace.execution_index),
    )
    swapped[raw_index] = replace(
        raw,
        trace=replace(raw.trace, execution_index=current.trace.execution_index),
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=tuple(swapped))

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("process_or_assignment",)
    assert result.summary is None


def test_foreign_graph_ids_are_derived_for_each_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = _model_leakage_sentinels()
    forged = tuple(
        replace(
            sentinel,
            foreign_release_id=sentinels[0].foreign_release_id,
            foreign_evidence_id=sentinels[0].foreign_evidence_id,
        )
        for sentinel in sentinels
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(leakage_sentinels=forged)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("leakage_sentinel",)
    assert result.summary is None


def test_self_consistent_forged_treatment_ids_fail_parent_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _model_outcomes(_strict_model_response)
    current = next(
        outcome
        for outcome in outcomes
        if (outcome.case_index, outcome.arm) == (0, "current_release")
    )
    forged_entries = tuple(
        replace(
            entry,
            revision_id=f"forged-{entry.revision_id}",
            candidate_id=f"forged-{entry.candidate_id}",
            evidence_ids=tuple(f"forged-{value}" for value in entry.evidence_ids),
        )
        for entry in current.trace.entries
    )
    revision_ids = tuple(entry.revision_id for entry in forged_entries)
    evidence_ids = tuple(
        value for entry in forged_entries for value in entry.evidence_ids
    )
    forged_trace = replace(
        current.trace,
        eligible_revision_ids=revision_ids,
        retrieved_revision_ids=revision_ids,
        returned_revision_ids=revision_ids,
        injected_revision_ids=revision_ids,
        source_evidence_ids=evidence_ids,
        entries=forged_entries,
        reader_audit=_model_audit(
            scope=current.trace.scope,
            source_kind="release",
            release_id=current.trace.release_id,
            entries=forged_entries,
        ),
    )
    forged = tuple(
        replace(outcome, trace=forged_trace) if outcome is current else outcome
        for outcome in outcomes
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=forged)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("treatment_contract",)
    assert result.summary is None


def test_capture_pid_is_consistent_across_case_arms_and_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _replace_model_trace(
        _model_outcomes(_strict_model_response),
        0,
        "current_release",
        capture_pid=999_999,
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=outcomes)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("process_or_assignment",)
    assert result.summary is None


def test_derived_references_match_persisted_first_eight_cases(
    tmp_path: Path,
) -> None:
    for case_index in range(8):
        case = helpfulness.generate_case(case_index)

        derived = helpfulness.derive_case_database_references(case)
        persisted = helpfulness.build_case_database(
            case,
            tmp_path / f"case-{case_index:03d}.sqlite",
        )

        assert derived == persisted


def test_model_wire_round_trip_is_closed_deterministic_and_exact_typed() -> None:
    result = _analyze_model_fixture()
    identity = _model_manifest()[0]
    outcome = _model_outcomes(_strict_model_response)[0]
    attrition = helpfulness.ModelRunAttrition(
        case_index=0,
        arm=_model_arm_order("nonce-000")[0],
        reason="timeout",
        attempted=True,
    )

    for value in (identity, outcome, attrition, result):
        encoded = helpfulness.wire_dumps(value)

        assert encoded == helpfulness.wire_dumps(value)
        assert encoded.endswith("\n")
        assert helpfulness.wire_loads(encoded) == value
        assert (
            json.dumps(
                json.loads(encoded),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            == encoded
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("arm", "memory_off"), ("case_index", 63)],
)
def test_model_arm_outcome_wire_rejects_outer_trace_contradictions(
    field: str,
    value: object,
) -> None:
    outcome = _model_outcomes(_strict_model_response)[0]
    encoded = json.loads(helpfulness.wire_dumps(outcome))
    assert encoded["payload"][field] != value
    encoded["payload"][field] = value

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_loads(
            json.dumps(encoded, sort_keys=True, separators=(",", ":")) + "\n"
        )

    assert error.value.reason == "closed_schema"


@pytest.mark.parametrize(
    "mutation",
    [
        "same_old_and_current",
        "duplicate_key",
        "wrong_mask",
        "invalid_key",
        "invalid_value",
    ],
)
def test_model_case_identity_wire_reuses_analysis_case_schema(
    mutation: str,
) -> None:
    case = helpfulness.generate_case(0)
    if mutation == "same_old_and_current":
        case = replace(case, old_value=case.current_value)
    elif mutation == "duplicate_key":
        case = replace(
            case,
            shared_entries=(
                replace(case.shared_entries[0], key=case.target_key),
                *case.shared_entries[1:],
            ),
        )
    elif mutation == "wrong_mask":
        case = replace(case, masked_value="BAD")
    elif mutation == "invalid_key":
        case = replace(case, target_key="bad")
    else:
        assert mutation == "invalid_value"
        case = replace(case, old_value="bad")
    identity = helpfulness.ModelCaseIdentity(
        case=case,
        case_manifest_sha256=helpfulness.case_manifest_sha256(case),
        references=helpfulness.derive_case_database_references(case),
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_dumps(identity)

    assert error.value.reason == "closed_schema"


def test_model_wire_rejects_unknown_nonfinite_bool_and_numpy_scalars() -> None:
    import numpy as np

    result = _analyze_model_fixture()
    assert result.summary is not None
    encoded = json.loads(helpfulness.wire_dumps(result))
    encoded["payload"]["unknown"] = "forbidden"

    with pytest.raises(helpfulness.WireProtocolError) as unknown:
        helpfulness.wire_loads(
            json.dumps(encoded, sort_keys=True, separators=(",", ":")) + "\n"
        )
    assert unknown.value.reason == "closed_schema"

    malformed_points = (float("nan"), float("inf"), True, np.float64(1.0))
    for malformed in malformed_points:
        estimate = replace(result.summary.delta_help, point=malformed)
        summary = replace(result.summary, delta_help=estimate)
        malformed_result = replace(result, summary=summary)

        with pytest.raises(helpfulness.WireProtocolError) as scalar:
            helpfulness.wire_dumps(malformed_result)
        assert scalar.value.reason == "closed_schema"

    malformed_attrition = helpfulness.ModelRunAttrition(
        case_index=True,
        arm="current_release",
        reason="timeout",
        attempted=True,
    )
    with pytest.raises(helpfulness.WireProtocolError) as boolean_integer:
        helpfulness.wire_dumps(malformed_attrition)
    assert boolean_integer.value.reason == "closed_schema"


def test_malformed_nested_references_fail_closed_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _model_manifest()
    malformed_identity = replace(
        manifest[0],
        references=replace(manifest[0].references, capture=object()),
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(
        manifest=(malformed_identity, *manifest[1:]),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("closed_schema",)
    assert result.summary is None
    assert result.attrition == ()


def test_malformed_nested_model_case_fails_closed_before_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _model_manifest()
    malformed_case = replace(
        manifest[0].case,
        shared_entries=(object(),) * 4,
    )
    malformed_identity = replace(manifest[0], case=malformed_case)
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(
        manifest=(malformed_identity, *manifest[1:]),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("closed_schema",)
    assert result.summary is None
    assert result.attrition == ()


def test_malformed_attrition_is_not_reflected_in_typed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_numpy_import(monkeypatch)

    result = helpfulness._analyze_model_traces(
        manifest=_model_manifest(),
        outcomes=_model_outcomes(_strict_model_response),
        attrition=(object(),),
        leakage_sentinels=_model_leakage_sentinels(),
        frozen_case_manifest_sha256s=_frozen_model_hashes(),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("closed_schema",)
    assert result.summary is None
    assert result.attrition == ()
    assert helpfulness.wire_loads(helpfulness.wire_dumps(result)) == result


def test_masked_equivalence_can_fail_after_abstention_gate_passes() -> None:
    def six_masked_answers(case: helpfulness.CodebookCase, arm: str) -> str:
        if case.case_index < 6 and arm == "target_masked":
            return case.current_value
        return _strict_model_response(case, arm)

    result = _analyze_model_fixture(six_masked_answers)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("masked_control",)
    assert result.efficacy == "not-assessed"
    assert result.safety == "not-assessed"
    assert result.stale_susceptibility == "not-assessed"
    assert result.summary is not None
    assert result.summary.masked_abstention_rate.point == 58 / 64
    assert result.summary.delta_masked_off.point == 0.09375
    assert result.summary.delta_masked_off.ci_lower == 0.03125
    assert result.summary.delta_masked_off.ci_upper == 0.171875


def test_model_result_wire_rejects_semantic_contradictions() -> None:
    valid = _analyze_model_fixture()
    masked_invalid = _analyze_model_fixture(
        lambda case, arm: (
            case.masked_value
            if arm == "target_masked"
            else _strict_model_response(case, arm)
        )
    )
    loss = helpfulness.ModelRunAttrition(
        case_index=0,
        arm=_model_arm_order("nonce-000")[0],
        reason="timeout",
        attempted=True,
    )
    malformed_results = (
        replace(valid, attrition=(loss,)),
        replace(valid, efficacy="not-assessed"),
        replace(valid, efficacy="negative"),
        replace(
            valid,
            summary=replace(valid.summary, access_denial_count=1),
        ),
        replace(masked_invalid, invalid_reasons=("made_up",)),
        replace(masked_invalid, invalid_reasons=("receipt_mismatch",)),
        replace(masked_invalid, efficacy="helpful"),
    )

    for malformed in malformed_results:
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_dumps(malformed)
        assert error.value.reason == "closed_schema"


def test_model_wire_recomputes_bootstrap_and_cross_metric_invariants() -> None:
    valid = _analyze_model_fixture()
    assert valid.summary is not None
    forged_ci = replace(valid.summary.delta_help, ci_lower=-1.0, ci_upper=1.0)
    forged_ci_result = replace(
        valid,
        efficacy="null-inconclusive",
        summary=replace(valid.summary, delta_help=forged_ci),
    )
    nonbinary_strict = helpfulness.MetricEstimate(
        per_case_values=(0.5,) * 64,
        point=0.5,
        ci_lower=0.5,
        ci_upper=0.5,
    )
    nonbinary_result = replace(
        valid,
        summary=replace(valid.summary, strict_signature_rate=nonbinary_strict),
    )
    forged_gap = helpfulness.MetricEstimate(
        per_case_values=(1.0, -1.0, *(0.0,) * 62),
        point=0.0,
        ci_lower=-0.1,
        ci_upper=0.1,
    )
    forged_gap_result = replace(
        valid,
        summary=replace(valid.summary, oracle_gap=forged_gap),
    )
    forged_masked = helpfulness.MetricEstimate(
        per_case_values=(0.0,) * 64,
        point=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
    )
    forged_masked_summary = replace(
        valid.summary,
        masked_abstention_rate=forged_masked,
    )
    forged_masked_result = helpfulness.ModelEvaluationResult(
        validity="invalid",
        efficacy="not-assessed",
        safety="not-assessed",
        stale_susceptibility="not-assessed",
        invalid_reasons=("masked_control",),
        summary=forged_masked_summary,
        attrition=(),
    )

    for malformed in (
        forged_ci_result,
        nonbinary_result,
        forged_gap_result,
        forged_masked_result,
    ):
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_dumps(malformed)
        assert error.value.reason == "closed_schema"


def test_model_wire_rejects_coverage_utility_and_attrition_forgery() -> None:
    valid = _analyze_model_fixture()
    assert valid.summary is not None
    zero = helpfulness.MetricEstimate(
        per_case_values=(0.0,) * 64,
        point=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
    )
    current = valid.summary.arm_summaries[0]
    forged_current = replace(
        current,
        assigned_target_coverage=zero,
        returned_target_coverage=zero,
        injected_target_coverage=zero,
    )
    coverage_forgery = replace(
        valid,
        summary=replace(
            valid.summary,
            arm_summaries=(forged_current, *valid.summary.arm_summaries[1:]),
        ),
    )
    half_delta = helpfulness.MetricEstimate(
        per_case_values=(0.5,) * 64,
        point=0.5,
        ci_lower=0.5,
        ci_upper=0.5,
    )
    utility_forgery = replace(
        valid,
        summary=replace(valid.summary, delta_raw=half_delta),
    )
    wrong_metric_type = replace(
        valid,
        summary=replace(valid.summary, delta_help=object()),
    )
    loss = helpfulness.ModelRunAttrition(
        case_index=0,
        arm=_model_arm_order("nonce-000")[0],
        reason="timeout",
        attempted=True,
    )
    duplicate_attrition = helpfulness.ModelEvaluationResult(
        validity="invalid",
        efficacy="not-assessed",
        safety="not-assessed",
        stale_susceptibility="not-assessed",
        invalid_reasons=("attrition",),
        summary=None,
        attrition=(loss, loss),
    )

    for malformed in (
        coverage_forgery,
        utility_forgery,
        wrong_metric_type,
        duplicate_attrition,
    ):
        with pytest.raises(helpfulness.WireProtocolError) as error:
            helpfulness.wire_dumps(malformed)
        assert error.value.reason == "closed_schema"


def test_model_result_wire_requires_canonical_attrition_order() -> None:
    losses = tuple(
        helpfulness.ModelRunAttrition(
            case_index=case_index,
            arm=_model_arm_order(f"nonce-{case_index:03d}")[0],
            reason="timeout",
            attempted=True,
        )
        for case_index in range(2)
    )
    reversed_result = helpfulness.ModelEvaluationResult(
        validity="invalid",
        efficacy="not-assessed",
        safety="not-assessed",
        stale_susceptibility="not-assessed",
        invalid_reasons=("attrition",),
        summary=None,
        attrition=tuple(reversed(losses)),
    )

    with pytest.raises(helpfulness.WireProtocolError) as error:
        helpfulness.wire_dumps(reversed_result)

    assert error.value.reason == "closed_schema"


@pytest.mark.parametrize("field", ["future_session_id", "future_run_id"])
def test_leakage_probe_identity_is_disjoint_from_outcomes(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _model_outcomes(_strict_model_response)
    sentinels = _model_leakage_sentinels()
    collision = getattr(outcomes[0].trace, field)
    malformed = (
        replace(sentinels[0], **{field: collision}),
        *sentinels[1:],
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(
        outcomes=outcomes,
        leakage_sentinels=malformed,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("process_or_assignment",)
    assert result.summary is None


@pytest.mark.parametrize(
    ("field", "collision"),
    [
        ("future_session_id", "nonce-000-capture-old"),
        ("future_run_id", "nonce-000-run-old"),
    ],
)
def test_future_identity_is_disjoint_from_local_capture(
    field: str,
    collision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _model_outcomes(_strict_model_response)
    current = next(
        outcome
        for outcome in outcomes
        if (outcome.case_index, outcome.arm) == (0, "current_release")
    )
    malformed_trace = replace(current.trace, **{field: collision})
    malformed = tuple(
        replace(outcome, trace=malformed_trace) if outcome is current else outcome
        for outcome in outcomes
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(outcomes=malformed)

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("process_or_assignment",)
    assert result.summary is None


@pytest.mark.parametrize("collision_owner", ["outcome", "probe"])
@pytest.mark.parametrize(
    ("field", "collision"),
    [
        ("future_session_id", "nonce-000-capture-foreign"),
        ("future_run_id", "nonce-000-run-foreign"),
    ],
)
def test_future_identity_is_disjoint_from_foreign_capture(
    collision_owner: str,
    field: str,
    collision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = _model_outcomes(_strict_model_response)
    sentinels = _model_leakage_sentinels()
    if collision_owner == "outcome":
        current = next(
            outcome
            for outcome in outcomes
            if (outcome.case_index, outcome.arm) == (0, "current_release")
        )
        malformed_trace = replace(current.trace, **{field: collision})
        outcomes = tuple(
            replace(outcome, trace=malformed_trace) if outcome is current else outcome
            for outcome in outcomes
        )
    else:
        sentinels = (
            replace(sentinels[0], **{field: collision}),
            *sentinels[1:],
        )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_model_fixture(
        outcomes=outcomes,
        leakage_sentinels=sentinels,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("process_or_assignment",)
    assert result.summary is None


def test_model_prompt_uses_frozen_production_byte_grammar() -> None:
    case = helpfulness.generate_case(0)
    context = b"context-bytes"
    query = _query(case)

    rendered = helpfulness.compose_model_prompt(context, query)

    prefix = b"[system]\n" + helpfulness.MODEL_SYSTEM_PROMPT + b"\n[memory]\n"
    assert rendered.prompt == prefix + context + b"[query]\n" + query + b"\n"
    assert rendered.context_start == len(prefix)
    assert rendered.context_end == len(prefix) + len(context)
    assert rendered.prompt[rendered.context_start : rendered.context_end] == context
    assert rendered.prompt.count(context) == 1


class _ByteModelTokenizer:
    def encode(
        self,
        value: bytes,
        *,
        add_special_tokens: bool,
    ) -> tuple[int, ...]:
        if not value and not add_special_tokens:
            return ()
        marker = 257 if add_special_tokens else 256
        return (marker, *value)


class _OneGateMismatchTokenizer:
    """Perturb exactly one preregistration invariant for mutation sensitivity."""

    def __init__(self, case: helpfulness.CodebookCase, gate: str) -> None:
        self.case = case
        self.gate = gate
        self.contexts = {
            arm: helpfulness._model_preregistration_context(case, arm)
            for arm in MODEL_ARMS
            if arm != "memory_off"
        }
        query = _query(case)
        self.prompts = {
            arm: helpfulness.compose_model_prompt(context, query).prompt
            for arm, context in self.contexts.items()
        }
        self.current_oracle_prompt_calls = 0

    def encode(
        self,
        value: bytes,
        *,
        add_special_tokens: bool,
    ) -> tuple[int, ...]:
        if not value and not add_special_tokens:
            return ()
        if add_special_tokens:
            if self.gate == "prompt_count" and value == self.prompts["stale_release"]:
                return (10, 11, 12)
            if (
                self.gate == "current_oracle_ids"
                and value == self.prompts["current_release"]
            ):
                self.current_oracle_prompt_calls += 1
                if self.current_oracle_prompt_calls % 2:
                    return (10, 11)
                return (11, 10)
            return (10, 11)
        if self.gate == "context_count" and value == self.contexts["stale_release"]:
            return (20, 21, 22)
        if self.gate == "value_count":
            if value == self.case.current_value.encode("ascii"):
                return (30,)
            if value == self.case.old_value.encode("ascii"):
                return (30, 31)
        return (20, 21)


class _RecordingModelBoundary:
    def __init__(self, response: object = "UNKNOWN", *, receipt: bool = True) -> None:
        self.response = response
        self.include_receipt = receipt
        self.received_token_ids: tuple[int, ...] | None = None
        self.received_prepared: helpfulness.PreparedModelCall | None = None

    def submit(
        self,
        input_token_ids: tuple[int, ...],
        *,
        prepared_call: helpfulness.PreparedModelCall,
    ) -> helpfulness.ModelBoundaryOutput:
        self.received_token_ids = input_token_ids
        self.received_prepared = prepared_call
        receipt = (
            helpfulness.make_model_call_receipt(
                submitted_prompt=prepared_call.prompt,
                context_start=prepared_call.context_start,
                context_end=prepared_call.context_end,
                input_token_ids=input_token_ids,
            )
            if self.include_receipt
            else None
        )
        return helpfulness.ModelBoundaryOutput(
            response=self.response,  # type: ignore[arg-type]
            receipt=receipt,
        )


def test_prepared_model_call_owns_receipts_and_submits_same_token_tuple() -> None:
    case = helpfulness.generate_case(0)
    context = b"context-bytes"
    query = _query(case)
    prepared = helpfulness.prepare_model_call(
        context,
        query,
        _ByteModelTokenizer(),
    )
    boundary = _RecordingModelBoundary()

    result = helpfulness.submit_model_call(prepared, boundary)

    assert result.valid is True
    assert result.invalid_reason is None
    assert result.response == "UNKNOWN"
    assert boundary.received_token_ids is prepared.input_token_ids
    assert boundary.received_prepared is prepared
    assert result.model_call_receipt == prepared.expected_receipt
    assert result.consumer_input_receipt == prepared.consumer_input_receipt
    consumer_receipt = prepared.consumer_input_receipt
    assert type(consumer_receipt) is helpfulness.ConsumerInputReceipt
    assert type(consumer_receipt.received_context_sha256) is str
    assert consumer_receipt.received_context_sha256 == sha256(context).hexdigest()
    assert type(consumer_receipt.received_context_utf8_bytes) is int
    assert consumer_receipt.received_context_utf8_bytes == len(context)
    assert type(consumer_receipt.received_query_sha256) is str
    assert consumer_receipt.received_query_sha256 == sha256(query).hexdigest()
    assert type(consumer_receipt.received_history_length) is int
    assert consumer_receipt.received_history_length == 0
    assert result.rendered_context_token_count == len(context) + 1
    assert prepared.expected_receipt.submitted_input_token_count == len(
        prepared.input_token_ids
    )


def test_receiptless_model_boundary_is_invalid_without_metadata_fallback() -> None:
    case = helpfulness.generate_case(0)
    prepared = helpfulness.prepare_model_call(
        b"context", _query(case), _ByteModelTokenizer()
    )

    result = helpfulness.submit_model_call(
        prepared,
        _RecordingModelBoundary(receipt=False),
    )

    assert result.valid is False
    assert result.invalid_reason == "model_call_receipt"
    assert result.model_call_receipt is None


def test_forged_model_boundary_receipt_is_invalid_without_metadata_fallback() -> None:
    prepared = helpfulness.prepare_model_call(
        b"context",
        _query(helpfulness.generate_case(0)),
        _ByteModelTokenizer(),
    )

    class ForgedReceiptBoundary(_RecordingModelBoundary):
        def submit(
            self,
            input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            del input_token_ids
            return helpfulness.ModelBoundaryOutput(
                response="UNKNOWN",
                receipt=replace(
                    prepared_call.expected_receipt,
                    submitted_prompt_sha256="0" * 64,
                ),
            )

    result = helpfulness.submit_model_call(prepared, ForgedReceiptBoundary())

    assert result.valid is False
    assert result.invalid_reason == "model_call_receipt"


def test_equal_comparing_wrong_receipt_type_is_invalid() -> None:
    prepared = helpfulness.prepare_model_call(
        b"context",
        _query(helpfulness.generate_case(0)),
        _ByteModelTokenizer(),
    )

    class AlwaysEqualReceipt:
        def __eq__(self, _other: object) -> bool:
            return True

    class WrongReceiptBoundary:
        def submit(
            self,
            _input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            del prepared_call
            return helpfulness.ModelBoundaryOutput(
                response="UNKNOWN",
                receipt=AlwaysEqualReceipt(),  # type: ignore[arg-type]
            )

    result = helpfulness.submit_model_call(prepared, WrongReceiptBoundary())

    assert result.valid is False
    assert result.invalid_reason == "model_call_receipt"
    assert result.model_call_receipt is None


@pytest.mark.parametrize(
    "field_name",
    [
        "submitted_prompt_context_start",
        "submitted_prompt_context_end",
        "submitted_input_token_count",
    ],
)
def test_model_boundary_receipt_integer_fields_require_exact_int(
    field_name: str,
) -> None:
    prepared = helpfulness.prepare_model_call(
        b"context",
        _query(helpfulness.generate_case(0)),
        _ByteModelTokenizer(),
    )

    class IntSubclass(int):
        pass

    wrong_value = IntSubclass(getattr(prepared.expected_receipt, field_name))

    class WrongFieldBoundary:
        def submit(
            self,
            _input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            return helpfulness.ModelBoundaryOutput(
                response="UNKNOWN",
                receipt=replace(
                    prepared_call.expected_receipt,
                    **{field_name: wrong_value},
                ),
            )

    result = helpfulness.submit_model_call(prepared, WrongFieldBoundary())

    assert result.valid is False
    assert result.model_call_receipt is None


@pytest.mark.parametrize(
    "field_name",
    [
        "submitted_prompt_sha256",
        "submitted_prompt_context_sha256",
        "submitted_input_token_ids_sha256",
    ],
)
def test_model_boundary_receipt_hash_fields_require_exact_str(
    field_name: str,
) -> None:
    prepared = helpfulness.prepare_model_call(
        b"context",
        _query(helpfulness.generate_case(0)),
        _ByteModelTokenizer(),
    )

    class ReceiptHash(str):
        pass

    class WrongHashBoundary:
        def submit(
            self,
            _input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            return helpfulness.ModelBoundaryOutput(
                response="UNKNOWN",
                receipt=replace(
                    prepared_call.expected_receipt,
                    **{
                        field_name: ReceiptHash(
                            getattr(prepared_call.expected_receipt, field_name)
                        )
                    },
                ),
            )

    result = helpfulness.submit_model_call(prepared, WrongHashBoundary())

    assert result.valid is False
    assert result.model_call_receipt is None


def test_manifest_rejects_systematically_wrong_consumer_receipt_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_receipt = helpfulness.ConsumerInputReceipt(
        received_context_sha256="0" * 64,
        received_context_utf8_bytes=999,
        received_query_sha256="0" * 64,
        received_history_length=0,
    )
    monkeypatch.setattr(
        helpfulness,
        "_expected_consumer_receipt",
        lambda _context, _query_bytes: wrong_receipt,
    )

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.prepare_model_run_manifest(
            _ByteModelTokenizer(),
            generator_commit_sha="a" * 40,
            evaluator_commit_sha="b" * 40,
            model_id="dry-run-model",
            model_weights_sha256="c" * 64,
            tokenizer_id="byte-tokenizer-v1",
            tokenizer_sha256="d" * 64,
        )

    assert type(error.value) is helpfulness.ModelProtocolError
    assert error.value.reason == "manifest_completeness"


@pytest.mark.parametrize(
    "field_name",
    [
        "received_context_sha256",
        "received_context_utf8_bytes",
        "received_query_sha256",
        "received_history_length",
    ],
)
def test_manifest_rejects_systematically_wrong_consumer_receipt_exact_type(
    field_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StrSubclass(str):
        pass

    class IntSubclass(int):
        pass

    def wrong_receipt(
        rendered_context: bytes,
        query: bytes,
    ) -> helpfulness.ConsumerInputReceipt:
        receipt = helpfulness.ConsumerInputReceipt(
            received_context_sha256=sha256(rendered_context).hexdigest(),
            received_context_utf8_bytes=len(rendered_context),
            received_query_sha256=sha256(query).hexdigest(),
            received_history_length=0,
        )
        value = getattr(receipt, field_name)
        wrong_value = StrSubclass(value) if type(value) is str else IntSubclass(value)
        return replace(receipt, **{field_name: wrong_value})

    monkeypatch.setattr(helpfulness, "_expected_consumer_receipt", wrong_receipt)

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.prepare_model_run_manifest(
            _ByteModelTokenizer(),
            generator_commit_sha="a" * 40,
            evaluator_commit_sha="b" * 40,
            model_id="dry-run-model",
            model_weights_sha256="c" * 64,
            tokenizer_id="byte-tokenizer-v1",
            tokenizer_sha256="d" * 64,
        )

    assert type(error.value) is helpfulness.ModelProtocolError
    assert error.value.reason == "manifest_completeness"


@pytest.mark.parametrize("token_ids", [(), (-1,), (True,)])
def test_model_tokenizer_rejects_empty_negative_and_bool_ids(
    token_ids: tuple[object, ...],
) -> None:
    class InvalidTokenizer:
        def encode(
            self,
            _value: bytes,
            *,
            add_special_tokens: bool,
        ) -> tuple[object, ...]:
            del add_special_tokens
            return token_ids

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.prepare_model_call(
            b"context",
            _query(helpfulness.generate_case(0)),
            InvalidTokenizer(),  # type: ignore[arg-type]
        )

    assert error.value.reason == "tokenizer_failure"


def test_model_boundary_rejects_non_string_response() -> None:
    prepared = helpfulness.prepare_model_call(
        b"context",
        _query(helpfulness.generate_case(0)),
        _ByteModelTokenizer(),
    )

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.submit_model_call(
            prepared,
            _RecordingModelBoundary(response=object()),
        )

    assert type(error.value) is helpfulness.ModelBoundaryExecutionError
    assert error.value.reason == "boundary_response"


def test_model_case_registration_freezes_arm_order_and_token_parity() -> None:
    result = helpfulness.prepare_model_case_registration(0, _ByteModelTokenizer())

    assert result.failure is None
    assert result.registration is not None
    registration = result.registration
    assert registration.model_attempt == 0
    assert tuple(call.arm for call in registration.arm_calls) == (
        "oracle",
        "current_release",
        "raw_history",
        "memory_off",
        "stale_release",
        "target_masked",
    )
    by_arm = {call.arm: call for call in registration.arm_calls}
    balanced_arms = set(MODEL_ARMS) - {"memory_off"}
    assert {
        by_arm[arm].prepared_call.rendered_context_token_count for arm in balanced_arms
    } == {registration.balanced_context_token_count}
    assert {
        len(by_arm[arm].prepared_call.input_token_ids) for arm in balanced_arms
    } == {registration.balanced_prompt_token_count}
    assert (
        registration.current_value_token_count == registration.stale_value_token_count
    )
    assert (
        by_arm["current_release"].prepared_call.prompt
        == by_arm["oracle"].prepared_call.prompt
    )
    assert (
        by_arm["current_release"].prepared_call.input_token_ids
        == by_arm["oracle"].prepared_call.input_token_ids
    )
    assert (
        by_arm["current_release"].prepared_call.expected_receipt
        == by_arm["oracle"].prepared_call.expected_receipt
    )


def test_memory_off_has_zero_context_tokens_but_nonempty_prompt_tokens() -> None:
    result = helpfulness.prepare_model_case_registration(0, _ByteModelTokenizer())

    assert result.failure is None
    assert result.registration is not None
    memory_off = next(
        call for call in result.registration.arm_calls if call.arm == "memory_off"
    )
    assert memory_off.rendered_context_utf8_bytes == 0
    assert memory_off.prepared_call.rendered_context_token_count == 0
    assert memory_off.prepared_call.input_token_ids


def test_model_candidate_rejects_only_context_token_count_mismatch() -> None:
    case = helpfulness.generate_case(0)
    tokenizer = _OneGateMismatchTokenizer(case, "context_count")

    assert (
        helpfulness._model_candidate_balance(case, tokenizer, model_attempt=0) is None
    )


def test_model_candidate_rejects_only_full_prompt_token_count_mismatch() -> None:
    case = helpfulness.generate_case(0)
    tokenizer = _OneGateMismatchTokenizer(case, "prompt_count")

    assert (
        helpfulness._model_candidate_balance(case, tokenizer, model_attempt=0) is None
    )


def test_model_candidate_rejects_only_standalone_value_token_count_mismatch() -> None:
    case = helpfulness.generate_case(0)
    tokenizer = _OneGateMismatchTokenizer(case, "value_count")

    assert (
        helpfulness._model_candidate_balance(case, tokenizer, model_attempt=0) is None
    )


def test_model_candidate_rejects_equal_length_current_oracle_token_id_mismatch() -> (
    None
):
    case = helpfulness.generate_case(0)
    balance_tokenizer = _OneGateMismatchTokenizer(case, "current_oracle_ids")
    assert (
        helpfulness._model_candidate_balance(
            case,
            balance_tokenizer,
            model_attempt=0,
        )
        is not None
    )

    result = helpfulness.prepare_model_case_registration(
        0,
        _OneGateMismatchTokenizer(case, "current_oracle_ids"),
    )

    assert result.registration is None
    assert result.failure == helpfulness.ModelManifestFailure(
        case_index=0,
        reason="tokenizer_instability",
        attempted_model_candidates=1,
    )


def test_model_case_search_accepts_first_naturally_matching_candidate() -> None:
    class FirstCandidateIsUnbalanced(_ByteModelTokenizer):
        def encode(
            self,
            value: bytes,
            *,
            add_special_tokens: bool,
        ) -> tuple[int, ...]:
            token_ids = super().encode(
                value,
                add_special_tokens=add_special_tokens,
            )
            if b"project-xk527d" in value and b"GSPFA" in value:
                return (*token_ids, 999)
            return token_ids

    result = helpfulness.prepare_model_case_registration(
        0,
        FirstCandidateIsUnbalanced(),
    )

    assert result.failure is None
    assert result.registration is not None
    assert result.registration.model_attempt == 1
    assert result.registration.identity.case.target_key == "project-xhz92s"
    assert result.registration.identity.case.old_value == "EZDKA"
    assert result.registration.identity.case.current_value == "BD45A"
    assert result.registration.identity.case.padding_entry == helpfulness.CodebookEntry(
        "project-2rwk7w",
        "NCAJS",
    )
    assert result.registration.identity.case_manifest_sha256 == (
        "1d840c53233cc9fbfe2454b64798357e06f0fd9e14e74160f4b0f15fdeaebe26"
    )


def test_model_case_search_exhausts_exact_fixed_range_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = helpfulness.generate_case(0)
    attempts: list[int] = []

    def candidate(case_index: int, *, model_attempt: int):
        assert case_index == 0
        attempts.append(model_attempt)
        return baseline

    monkeypatch.setattr(helpfulness, "_generate_model_candidate", candidate)
    monkeypatch.setattr(
        helpfulness,
        "_model_candidate_balance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        helpfulness,
        "derive_case_database_references",
        lambda _case: pytest.fail("references derived before candidate acceptance"),
    )

    result = helpfulness.prepare_model_case_registration(0, _ByteModelTokenizer())

    assert result.registration is None
    assert result.failure == helpfulness.ModelManifestFailure(
        case_index=0,
        reason="candidate_exhausted",
        attempted_model_candidates=100_000,
    )
    assert attempts == list(range(100_000))


def test_model_candidate_attempt_formula_is_model_times_256_plus_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_token = helpfulness._token
    attempts: list[int] = []

    def recording_token(seed, case_index, label, item_index, attempt, length):
        attempts.append(attempt)
        return real_token(seed, case_index, label, item_index, attempt, length)

    monkeypatch.setattr(helpfulness, "_token", recording_token)

    case = helpfulness._generate_model_candidate(0, model_attempt=7)

    assert case is not None
    assert attempts
    assert all(7 * 256 <= attempt <= 7 * 256 + 255 for attempt in attempts)


def test_model_collision_uses_plus_one_and_resets_for_each_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_token = helpfulness._token
    model_attempt = 3
    base = model_attempt * 256
    target_token = real_token(helpfulness.CASE_SEED, 0, "target", 0, base, 6)
    observed: dict[tuple[str, int], list[int]] = {}

    def collide_shared_zero_once(seed, case_index, label, item_index, attempt, length):
        observed.setdefault((label, item_index), []).append(attempt)
        if label == "shared" and item_index == 0 and attempt == base:
            return target_token
        return real_token(seed, case_index, label, item_index, attempt, length)

    monkeypatch.setattr(helpfulness, "_token", collide_shared_zero_once)

    case = helpfulness._generate_model_candidate(0, model_attempt=model_attempt)

    assert case is not None
    assert observed[("shared", 0)][:2] == [base, base + 1]
    assert observed[("shared", 1)][0] == base


@pytest.mark.parametrize(
    ("last_collision_succeeds", "candidate_exists"),
    [(True, True), (False, False)],
)
def test_model_collision_lane_has_exact_256_attempt_boundary(
    last_collision_succeeds: bool,
    candidate_exists: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_token = helpfulness._token
    model_attempt = 4
    base = model_attempt * 256
    target_token = real_token(helpfulness.CASE_SEED, 0, "target", 0, base, 6)
    shared_zero_attempts: list[int] = []

    def collide_shared_zero(seed, case_index, label, item_index, attempt, length):
        if label == "shared" and item_index == 0:
            shared_zero_attempts.append(attempt)
            if not last_collision_succeeds or attempt < base + 255:
                return target_token
        return real_token(seed, case_index, label, item_index, attempt, length)

    monkeypatch.setattr(helpfulness, "_token", collide_shared_zero)

    case = helpfulness._generate_model_candidate(0, model_attempt=model_attempt)

    assert (case is not None) is candidate_exists
    assert shared_zero_attempts == list(range(base, base + 256))


def _build_prepared_model_manifest() -> helpfulness.ModelRunManifest:
    result = helpfulness.prepare_model_run_manifest(
        _ByteModelTokenizer(),
        generator_commit_sha="a" * 40,
        evaluator_commit_sha="b" * 40,
        model_id="dry-run-model",
        model_weights_sha256="c" * 64,
        tokenizer_id="byte-tokenizer-v1",
        tokenizer_sha256="d" * 64,
    )
    assert result.failure is None
    assert result.manifest is not None
    return result.manifest


@pytest.fixture(scope="module")
def prepared_model_manifest() -> helpfulness.ModelRunManifest:
    """Share only the recursively frozen manifest; tokenizers remain test-local."""

    return _build_prepared_model_manifest()


def _run_registered_strict_model(
    manifest: helpfulness.ModelRunManifest,
) -> helpfulness.ModelDryRunResult:
    responses = {
        id(arm_call.prepared_call): _strict_model_response(
            registration.identity.case,
            arm_call.arm,
        )
        for registration in manifest.cases
        for arm_call in registration.arm_calls
    }

    class StrictBoundary:
        def submit(
            self,
            input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            return helpfulness.ModelBoundaryOutput(
                response=responses[id(prepared_call)],
                receipt=helpfulness.make_model_call_receipt(
                    submitted_prompt=prepared_call.prompt,
                    context_start=prepared_call.context_start,
                    context_end=prepared_call.context_end,
                    input_token_ids=input_token_ids,
                ),
            )

    return helpfulness.run_model_dry_run(
        manifest,
        _ByteModelTokenizer(),
        StrictBoundary(),
    )


def _registered_model_outcomes(
    manifest: helpfulness.ModelRunManifest,
    dry_run: helpfulness.ModelDryRunResult,
) -> tuple[helpfulness.ModelArmOutcome, ...]:
    outcomes: list[helpfulness.ModelArmOutcome] = []
    call_index = 0
    for case_index, registration in enumerate(manifest.cases):
        case = registration.identity.case
        for execution_offset, arm_call in enumerate(registration.arm_calls):
            dry_call = dry_run.calls[call_index]
            call_index += 1
            execution = dry_call.execution
            assert dry_call.case_index == case_index
            assert dry_call.arm == arm_call.arm
            assert execution is not None and execution.valid
            receipt = execution.model_call_receipt
            assert receipt is not None
            consumer = execution.consumer_input_receipt
            trace = replace(
                _model_trace(case, arm_call.arm, execution.response),
                execution_index=case_index * len(MODEL_ARMS) + execution_offset,
                rendered_context_sha256=arm_call.rendered_context_sha256,
                rendered_context_utf8_bytes=arm_call.rendered_context_utf8_bytes,
                rendered_context_token_count=execution.rendered_context_token_count,
                received_context_sha256=consumer.received_context_sha256,
                received_context_utf8_bytes=consumer.received_context_utf8_bytes,
                received_query_sha256=consumer.received_query_sha256,
                submitted_prompt_sha256=receipt.submitted_prompt_sha256,
                submitted_prompt_context_start=(receipt.submitted_prompt_context_start),
                submitted_prompt_context_end=receipt.submitted_prompt_context_end,
                submitted_prompt_context_sha256=(
                    receipt.submitted_prompt_context_sha256
                ),
                submitted_input_token_ids_sha256=(
                    receipt.submitted_input_token_ids_sha256
                ),
                submitted_input_token_count=receipt.submitted_input_token_count,
                query_sha256=registration.query_sha256,
                history_length=consumer.received_history_length,
                response=execution.response,
            )
            outcomes.append(
                helpfulness.ModelArmOutcome(
                    case_index=case_index,
                    arm=arm_call.arm,
                    trace=trace,
                )
            )
    assert call_index == len(dry_run.calls)
    return tuple(outcomes)


@pytest.fixture(scope="module")
def registered_model_evidence(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> tuple[
    helpfulness.ModelDryRunResult,
    tuple[helpfulness.ModelArmOutcome, ...],
    tuple[helpfulness.LeakageSentinelTrace, ...],
]:
    dry_run = _run_registered_strict_model(prepared_model_manifest)
    cases = tuple(
        registration.identity.case for registration in prepared_model_manifest.cases
    )
    return (
        dry_run,
        _registered_model_outcomes(prepared_model_manifest, dry_run),
        _model_leakage_sentinels(cases),
    )


def _analyze_registered_fixture(
    manifest: helpfulness.ModelRunManifest,
    evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    *,
    dry_run: helpfulness.ModelDryRunResult | None = None,
    outcomes: tuple[helpfulness.ModelArmOutcome, ...] | None = None,
) -> helpfulness.ModelEvaluationResult:
    frozen_dry_run, frozen_outcomes, leakage_sentinels = evidence
    return helpfulness.analyze_model_run(
        manifest=manifest,
        tokenizer=_ByteModelTokenizer(),
        dry_run=frozen_dry_run if dry_run is None else dry_run,
        outcomes=frozen_outcomes if outcomes is None else outcomes,
        leakage_sentinels=leakage_sentinels,
    )


def test_public_model_analysis_has_one_manifest_root_and_scores_bound_evidence(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
) -> None:
    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
    )

    assert tuple(inspect.signature(helpfulness.analyze_model_run).parameters) == (
        "manifest",
        "tokenizer",
        "dry_run",
        "outcomes",
        "leakage_sentinels",
    )
    assert result.validity == "valid"
    assert result.efficacy == "helpful"
    assert result.safety == "non-increased"
    assert result.stale_susceptibility == "stale-sensitive"


def test_public_model_analysis_rejects_unregistered_prompt_receipts(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        outcomes=_model_outcomes(_strict_model_response),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("receipt_mismatch",)
    assert result.summary is None


def test_public_model_analysis_rejects_dry_run_from_another_manifest(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, _outcomes, _sentinels = registered_model_evidence
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(dry_run, manifest_sha256="0" * 64),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("manifest_mismatch",)
    assert result.summary is None


def test_public_model_analysis_rejects_reordered_dry_run_slots(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, _outcomes, _sentinels = registered_model_evidence
    reordered = replace(
        dry_run,
        calls=(dry_run.calls[1], dry_run.calls[0], *dry_run.calls[2:]),
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=reordered,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("execution_completeness",)
    assert result.summary is None


def test_public_model_analysis_cross_binds_response_to_call_result(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, _outcomes, _sentinels = registered_model_evidence
    execution = dry_run.calls[0].execution
    assert execution is not None
    forged_call = replace(
        dry_run.calls[0],
        execution=replace(execution, response="FABRICATED-AFTER-THE-CALL"),
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(dry_run, calls=(forged_call, *dry_run.calls[1:])),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("receipt_mismatch",)
    assert result.summary is None


def test_public_model_analysis_cross_binds_trace_to_registered_receipt(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dry_run, outcomes, _sentinels = registered_model_evidence
    forged = (
        replace(
            outcomes[0],
            trace=replace(
                outcomes[0].trace,
                submitted_prompt_sha256="0" * 64,
            ),
        ),
        *outcomes[1:],
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        outcomes=forged,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("receipt_mismatch",)
    assert result.summary is None


def test_public_model_analysis_rejects_jointly_forged_dry_and_trace_receipt(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, outcomes, _sentinels = registered_model_evidence
    execution = dry_run.calls[0].execution
    assert execution is not None
    receipt = execution.model_call_receipt
    assert receipt is not None
    forged_receipt = replace(receipt, submitted_prompt_sha256="0" * 64)
    forged_call = replace(
        dry_run.calls[0],
        execution=replace(execution, model_call_receipt=forged_receipt),
    )
    forged_outcome = replace(
        outcomes[0],
        trace=replace(
            outcomes[0].trace,
            submitted_prompt_sha256=forged_receipt.submitted_prompt_sha256,
        ),
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(dry_run, calls=(forged_call, *dry_run.calls[1:])),
        outcomes=(forged_outcome, *outcomes[1:]),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("model_call_receipt",)
    assert result.summary is None


@pytest.mark.parametrize("mutation", ["consumer_hash", "context_token_count"])
def test_public_model_analysis_rejects_jointly_forged_consumer_facts(
    mutation: str,
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, outcomes, _sentinels = registered_model_evidence
    execution = dry_run.calls[0].execution
    assert execution is not None
    if mutation == "consumer_hash":
        forged_consumer = replace(
            execution.consumer_input_receipt,
            received_context_sha256="0" * 64,
        )
        forged_execution = replace(
            execution,
            consumer_input_receipt=forged_consumer,
        )
        forged_trace = replace(
            outcomes[0].trace,
            received_context_sha256=forged_consumer.received_context_sha256,
        )
    else:
        forged_execution = replace(
            execution,
            rendered_context_token_count=execution.rendered_context_token_count + 1,
        )
        forged_trace = replace(
            outcomes[0].trace,
            rendered_context_token_count=forged_execution.rendered_context_token_count,
        )
    forged_call = replace(dry_run.calls[0], execution=forged_execution)
    forged_outcome = replace(outcomes[0], trace=forged_trace)
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(dry_run, calls=(forged_call, *dry_run.calls[1:])),
        outcomes=(forged_outcome, *outcomes[1:]),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("receipt_mismatch",)
    assert result.summary is None


def test_public_model_analysis_recomputes_invalid_call_ledger(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, _outcomes, _sentinels = registered_model_evidence
    first = dry_run.calls[0]
    forged_invalid = helpfulness.ModelDryRunInvalidCall(
        case_index=first.case_index,
        arm=first.arm,
        attempted=True,
        reason="model_call_failure",
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(
            dry_run,
            validity="invalid",
            invalid_calls=(forged_invalid,),
        ),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("execution_completeness",)
    assert result.summary is None


def test_public_model_analysis_rejects_missing_invalid_call_ledger_entry(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, outcomes, _sentinels = registered_model_evidence
    target = dry_run.calls[11]
    failed = replace(target, execution=None)
    remaining = (*outcomes[:11], *outcomes[12:])
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(
            dry_run,
            validity="invalid",
            calls=(*dry_run.calls[:11], failed, *dry_run.calls[12:]),
        ),
        outcomes=remaining,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("execution_completeness",)
    assert result.summary is None


def test_public_model_analysis_rejects_validity_flag_lie(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, _outcomes, _sentinels = registered_model_evidence
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(dry_run, validity="invalid"),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("closed_schema",)
    assert result.summary is None


@pytest.mark.parametrize("mutation", ["call_bool_index", "ledger_integer_bool"])
def test_public_model_analysis_classifies_dry_run_type_confusion_as_schema(
    mutation: str,
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, _outcomes, _sentinels = registered_model_evidence
    if mutation == "call_bool_index":
        malformed = replace(
            dry_run,
            calls=(replace(dry_run.calls[0], case_index=False), *dry_run.calls[1:]),
        )
    else:
        first = dry_run.calls[0]
        malformed = replace(
            dry_run,
            validity="invalid",
            invalid_calls=(
                helpfulness.ModelDryRunInvalidCall(
                    case_index=first.case_index,
                    arm=first.arm,
                    attempted=1,  # type: ignore[arg-type]
                    reason="model_call_failure",
                ),
            ),
        )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=malformed,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("closed_schema",)
    assert result.summary is None


def test_public_model_analysis_rejects_self_consistent_receiptless_call(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, outcomes, _sentinels = registered_model_evidence
    target = dry_run.calls[0]
    execution = target.execution
    assert execution is not None
    invalid_call = replace(
        target,
        execution=replace(
            execution,
            model_call_receipt=None,
            valid=False,
            invalid_reason="model_call_receipt",
        ),
    )
    invalid_receipt = helpfulness.ModelDryRunInvalidCall(
        case_index=target.case_index,
        arm=target.arm,
        attempted=True,
        reason="model_call_receipt",
    )
    malformed = replace(
        dry_run,
        validity="invalid",
        invalid_calls=(invalid_receipt,),
        calls=(invalid_call, *dry_run.calls[1:]),
    )
    remaining = tuple(
        outcome
        for outcome in outcomes
        if (outcome.case_index, outcome.arm) != (target.case_index, target.arm)
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=malformed,
        outcomes=remaining,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("model_call_receipt",)
    assert result.summary is None
    assert result.attrition == ()


def test_public_model_analysis_prioritizes_any_invalid_model_receipt(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, outcomes, _sentinels = registered_model_evidence
    target_index = 300
    target = dry_run.calls[target_index]
    execution = target.execution
    assert execution is not None
    receiptless = replace(
        target,
        execution=replace(
            execution,
            model_call_receipt=None,
            valid=False,
            invalid_reason="model_call_receipt",
        ),
    )
    invalid_receipt = helpfulness.ModelDryRunInvalidCall(
        case_index=target.case_index,
        arm=target.arm,
        attempted=True,
        reason="model_call_receipt",
    )
    earlier_mismatch = (
        replace(
            outcomes[0],
            trace=replace(outcomes[0].trace, response="EARLY-MISMATCH"),
        ),
        *outcomes[1:],
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(
            dry_run,
            validity="invalid",
            invalid_calls=(invalid_receipt,),
            calls=(
                *dry_run.calls[:target_index],
                receiptless,
                *dry_run.calls[target_index + 1 :],
            ),
        ),
        outcomes=earlier_mismatch,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("model_call_receipt",)
    assert result.summary is None


def test_public_model_analysis_rejects_outcome_for_failed_call(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dry_run, _outcomes, _sentinels = registered_model_evidence
    target = dry_run.calls[13]
    failure = helpfulness.ModelDryRunInvalidCall(
        case_index=target.case_index,
        arm=target.arm,
        attempted=True,
        reason="model_call_failure",
    )
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=replace(
            dry_run,
            validity="invalid",
            invalid_calls=(failure,),
            calls=(
                *dry_run.calls[:13],
                replace(target, execution=None),
                *dry_run.calls[14:],
            ),
        ),
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("execution_completeness",)
    assert result.summary is None


def test_public_model_analysis_derives_fixed_slot_boundary_attrition(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
) -> None:
    dry_run, outcomes, _sentinels = registered_model_evidence
    target = dry_run.calls[17]
    failed_call = replace(target, execution=None)
    failure = helpfulness.ModelDryRunInvalidCall(
        case_index=target.case_index,
        arm=target.arm,
        attempted=True,
        reason="model_call_failure",
    )
    failed_run = replace(
        dry_run,
        validity="invalid",
        invalid_calls=(failure,),
        calls=(*dry_run.calls[:17], failed_call, *dry_run.calls[18:]),
    )
    remaining = tuple(
        outcome
        for outcome in outcomes
        if (outcome.case_index, outcome.arm) != (target.case_index, target.arm)
    )

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        dry_run=failed_run,
        outcomes=remaining,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("attrition",)
    assert result.summary is None
    assert result.attrition == (
        helpfulness.ModelRunAttrition(
            case_index=target.case_index,
            arm=target.arm,
            reason="model_call_failure",
            attempted=True,
        ),
    )


def test_public_model_analysis_rejects_missing_successful_outcome(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
) -> None:
    _dry_run, outcomes, _sentinels = registered_model_evidence
    remaining = (*outcomes[:29], *outcomes[30:])

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        outcomes=remaining,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("execution_completeness",)
    assert result.summary is None
    assert result.attrition == ()


def test_public_model_analysis_does_not_silently_reorder_outcomes(
    prepared_model_manifest: helpfulness.ModelRunManifest,
    registered_model_evidence: tuple[
        helpfulness.ModelDryRunResult,
        tuple[helpfulness.ModelArmOutcome, ...],
        tuple[helpfulness.LeakageSentinelTrace, ...],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dry_run, outcomes, _sentinels = registered_model_evidence
    reordered = (outcomes[1], outcomes[0], *outcomes[2:])
    _forbid_numpy_import(monkeypatch)

    result = _analyze_registered_fixture(
        prepared_model_manifest,
        registered_model_evidence,
        outcomes=reordered,
    )

    assert result.validity == "invalid"
    assert result.invalid_reasons == ("execution_completeness",)
    assert result.summary is None


def test_full_model_manifest_is_canonical_complete_and_preregistered() -> None:
    # Keep one uncached construction as the deterministic end-to-end golden.
    manifest = _build_prepared_model_manifest()
    tokenizer = _ByteModelTokenizer()

    encoded = helpfulness.model_run_manifest_bytes(manifest, tokenizer)
    parsed = json.loads(encoded)
    assert (
        helpfulness.model_run_manifest_sha256(manifest, tokenizer)
        == sha256(encoded).hexdigest()
    )
    assert helpfulness.model_run_manifest_sha256(manifest, tokenizer) == (
        "31eac9a9e274fc282270863efb09a7a884ee784f57032f9224b2f0bba4570bbb"
    )
    assert len(manifest.cases) == 64
    assert tuple(case.identity.case.case_index for case in manifest.cases) == tuple(
        range(64)
    )
    assert parsed["generator_commit_sha"] == "a" * 40
    assert parsed["evaluator_commit_sha"] == "b" * 40
    assert parsed["model_id"] == "dry-run-model"
    assert parsed["model_weights_sha256"] == "c" * 64
    assert parsed["tokenizer_id"] == "byte-tokenizer-v1"
    assert parsed["tokenizer_sha256"] == "d" * 64
    assert parsed["system_prompt_sha256"] == (
        "2b3ec812307450d7e06806a62f240471d66aed192cb634873fa1397bac7cbaa7"
    )
    assert parsed["prompt_grammar_sha256"] == (
        "ada71fbb0b59baeb565858780ede298f9d2a0a0f8026fb8361500a371c59e4e4"
    )
    assert parsed["renderer_sha256"] == (
        "0656efbe3e6e107f2858cfa0dac99fee986048c2c8a031fb9f92205624bb4051"
    )
    assert parsed["query_template_sha256"] == (
        "a5c17c493e037c8bd3d43e35ba2f35875f5fe8eadbf2036c432f70e457a6ec50"
    )
    assert parsed["profile"] == "model-helpfulness-v1"
    assert parsed["case_seed"] == helpfulness.CASE_SEED
    assert parsed["case_count"] == 64
    assert parsed["call_count"] == 384
    assert parsed["decoding"] == {
        "mode": "greedy",
        "samples": 1,
        "temperature": "0",
    }
    assert parsed["bootstrap"] == {
        "algorithm": "paired-pcg64-percentile-linear-type7-v1",
        "matrix_sha256": helpfulness.MODEL_BOOTSTRAP_MATRIX_SHA256,
        "resamples": 10_000,
        "seed": 20_260_708,
    }
    assert all(type(item["value"]) is str for item in parsed["thresholds"])
    assert tuple(case["case_index"] for case in parsed["cases"]) == tuple(range(64))
    assert all(
        tuple(arm["arm"] for arm in case["arms"])
        == helpfulness.model_arm_order(case["case_id"])
        for case in parsed["cases"]
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_model_manifest_rejects_missing_extra_and_reordered_cases(
    mutation: str,
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    if mutation == "missing":
        cases = manifest.cases[:-1]
    elif mutation == "extra":
        cases = (*manifest.cases, manifest.cases[-1])
    else:
        cases = (manifest.cases[1], manifest.cases[0], *manifest.cases[2:])

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.model_run_manifest_bytes(
            replace(manifest, cases=cases),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_model_manifest_fails_closed_on_malformed_nested_identity(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    malformed_case = replace(manifest.cases[0], identity=object())

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.model_run_manifest_bytes(
            replace(manifest, cases=(malformed_case, *manifest.cases[1:])),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_model_manifest_rejects_bool_schema_version(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.model_run_manifest_bytes(
            replace(manifest, schema_version=True),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_model_manifest_rejects_equal_but_wrong_nested_threshold_type(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest

    class AlwaysEqual:
        def __eq__(self, _other: object) -> bool:
            return True

    malformed_thresholds = (AlwaysEqual(), *manifest.thresholds[1:])
    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.model_run_manifest_bytes(
            replace(
                manifest,
                thresholds=malformed_thresholds,  # type: ignore[arg-type]
            ),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_model_manifest_rejects_lie_about_first_matching_attempt(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    forged_case = replace(
        manifest.cases[0],
        model_attempt=manifest.cases[0].model_attempt + 1,
    )

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.validate_model_run_manifest(
            replace(manifest, cases=(forged_case, *manifest.cases[1:])),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_dry_run_replays_tokenizer_before_boundary_on_attempt_lie(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    forged_case = replace(manifest.cases[0], model_attempt=1)
    boundary_calls = 0

    class ForbiddenBoundary:
        def submit(self, *_args, **_kwargs):
            nonlocal boundary_calls
            boundary_calls += 1
            raise AssertionError("boundary must not run")

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.run_model_dry_run(
            replace(manifest, cases=(forged_case, *manifest.cases[1:])),
            _ByteModelTokenizer(),
            ForbiddenBoundary(),
        )

    assert error.value.reason == "manifest_completeness"
    assert boundary_calls == 0


def test_model_manifest_rejects_rehashed_same_length_raw_token_forgery(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    case = manifest.cases[0]
    raw_offset = next(
        offset
        for offset, call in enumerate(case.arm_calls)
        if call.arm == "raw_history"
    )
    raw = case.arm_calls[raw_offset]
    prepared = raw.prepared_call
    forged_token_ids = tuple(token_id + 1 for token_id in prepared.input_token_ids)
    forged_receipt = helpfulness.make_model_call_receipt(
        submitted_prompt=prepared.prompt,
        context_start=prepared.context_start,
        context_end=prepared.context_end,
        input_token_ids=forged_token_ids,
    )
    forged_raw = replace(
        raw,
        prepared_call=replace(
            prepared,
            input_token_ids=forged_token_ids,
            expected_receipt=forged_receipt,
        ),
    )
    forged_arms = tuple(
        forged_raw if offset == raw_offset else call
        for offset, call in enumerate(case.arm_calls)
    )
    forged_case = replace(case, arm_calls=forged_arms)

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.validate_model_run_manifest(
            replace(manifest, cases=(forged_case, *manifest.cases[1:])),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_model_manifest_rejects_bool_nested_consumer_history_length(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    case = manifest.cases[0]
    call = case.arm_calls[0]
    prepared = call.prepared_call
    forged_call = replace(
        call,
        prepared_call=replace(
            prepared,
            consumer_input_receipt=replace(
                prepared.consumer_input_receipt,
                received_history_length=False,
            ),
        ),
    )
    forged_case = replace(case, arm_calls=(forged_call, *case.arm_calls[1:]))

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.validate_model_run_manifest(
            replace(manifest, cases=(forged_case, *manifest.cases[1:])),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_model_manifest_rejects_str_subclass_threshold_field(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest

    class StrSubclass(str):
        pass

    first = manifest.thresholds[0]
    forged_threshold = helpfulness.ModelThreshold(
        name=StrSubclass(first.name),
        value=first.value,
    )

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.validate_model_run_manifest(
            replace(
                manifest,
                thresholds=(forged_threshold, *manifest.thresholds[1:]),
            ),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


@pytest.mark.parametrize(
    "public_gate",
    ["validate", "bytes", "sha256", "dry_run"],
)
def test_public_manifest_gates_preserve_replay_tokenizer_failure(
    public_gate: str,
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest

    class FirstEncodeFailsTokenizer(_ByteModelTokenizer):
        def __init__(self) -> None:
            self.calls = 0

        def encode(
            self,
            value: bytes,
            *,
            add_special_tokens: bool,
        ) -> tuple[int, ...]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("replay tokenizer failed")
            return super().encode(value, add_special_tokens=add_special_tokens)

    class CountingBoundary(_RecordingModelBoundary):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def submit(
            self,
            input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            self.calls += 1
            return super().submit(
                input_token_ids,
                prepared_call=prepared_call,
            )

    tokenizer = FirstEncodeFailsTokenizer()
    boundary = CountingBoundary()
    with pytest.raises(helpfulness.ModelProtocolError) as error:
        if public_gate == "validate":
            helpfulness.validate_model_run_manifest(manifest, tokenizer)
        elif public_gate == "bytes":
            helpfulness.model_run_manifest_bytes(manifest, tokenizer)
        elif public_gate == "sha256":
            helpfulness.model_run_manifest_sha256(manifest, tokenizer)
        else:
            helpfulness.run_model_dry_run(manifest, tokenizer, boundary)

    assert error.value.reason == "tokenizer_failure"
    assert tokenizer.calls == 1
    assert boundary.calls == 0


def test_replay_candidate_exhaustion_remains_manifest_completeness(
    monkeypatch: pytest.MonkeyPatch,
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    monkeypatch.setattr(helpfulness, "MODEL_ATTEMPT_LIMIT", 1)
    monkeypatch.setattr(
        helpfulness,
        "_model_candidate_balance",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.validate_model_run_manifest(manifest, _ByteModelTokenizer())

    assert error.value.reason == "manifest_completeness"


def test_model_attempt_zero_reproduces_original_generator_for_all_cases() -> None:
    assert tuple(
        helpfulness._generate_model_candidate(case_index, model_attempt=0)
        for case_index in range(64)
    ) == tuple(helpfulness.generate_case(case_index) for case_index in range(64))


def test_full_manifest_stops_at_midrun_exhaustion_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = helpfulness.prepare_model_case_registration
    visited: list[int] = []

    def fail_case_seven(case_index: int, tokenizer):
        visited.append(case_index)
        if case_index == 7:
            return helpfulness.ModelCaseRegistrationResult(
                registration=None,
                failure=helpfulness.ModelManifestFailure(
                    case_index=7,
                    reason="candidate_exhausted",
                    attempted_model_candidates=100_000,
                ),
            )
        return original(case_index, tokenizer)

    monkeypatch.setattr(
        helpfulness,
        "prepare_model_case_registration",
        fail_case_seven,
    )
    monkeypatch.setattr(
        helpfulness,
        "submit_model_call",
        lambda *_args, **_kwargs: pytest.fail("model boundary called during manifest"),
    )

    result = helpfulness.prepare_model_run_manifest(
        _ByteModelTokenizer(),
        generator_commit_sha="a" * 40,
        evaluator_commit_sha="b" * 40,
        model_id="dry-run-model",
        model_weights_sha256="c" * 64,
        tokenizer_id="byte-tokenizer-v1",
        tokenizer_sha256="d" * 64,
    )

    assert result.manifest is None
    assert result.failure == helpfulness.ModelManifestFailure(
        case_index=7,
        reason="candidate_exhausted",
        attempted_model_candidates=100_000,
    )
    assert visited == list(range(8))


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reordered"])
def test_model_manifest_rejects_incomplete_or_reordered_arm_schedule(
    mutation: str,
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    case = manifest.cases[0]
    if mutation == "missing":
        arms = case.arm_calls[:-1]
    elif mutation == "extra":
        arms = (*case.arm_calls, case.arm_calls[0])
    elif mutation == "duplicate":
        arms = (case.arm_calls[0], case.arm_calls[0], *case.arm_calls[2:])
    else:
        arms = (case.arm_calls[1], case.arm_calls[0], *case.arm_calls[2:])
    forged_case = replace(case, arm_calls=arms)

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.validate_model_run_manifest(
            replace(manifest, cases=(forged_case, *manifest.cases[1:])),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_model_manifest_rejects_nested_receipt_forgery(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    case = manifest.cases[0]
    call = case.arm_calls[0]
    forged_call = replace(
        call,
        prepared_call=replace(
            call.prepared_call,
            expected_receipt=replace(
                call.prepared_call.expected_receipt,
                submitted_prompt_sha256="0" * 64,
            ),
        ),
    )
    forged_case = replace(case, arm_calls=(forged_call, *case.arm_calls[1:]))

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.validate_model_run_manifest(
            replace(manifest, cases=(forged_case, *manifest.cases[1:])),
            _ByteModelTokenizer(),
        )

    assert error.value.reason == "manifest_completeness"


def test_dry_run_validates_complete_manifest_before_boundary(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    boundary = _RecordingModelBoundary()

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.run_model_dry_run(
            replace(manifest, cases=manifest.cases[:-1]),
            _ByteModelTokenizer(),
            boundary,
        )

    assert error.value.reason == "manifest_completeness"
    assert boundary.received_token_ids is None


def test_dry_run_submits_all_registered_calls_without_expected_answer(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    boundary = _RecordingModelBoundary()

    tokenizer = _ByteModelTokenizer()
    result = helpfulness.run_model_dry_run(manifest, tokenizer, boundary)

    assert result.validity == "valid"
    assert result.invalid_calls == ()
    assert result.manifest_sha256 == helpfulness.model_run_manifest_sha256(
        manifest,
        tokenizer,
    )
    assert len(result.calls) == 64 * 6
    assert all(call.execution.valid for call in result.calls)
    assert tuple(call.arm for call in result.calls[:6]) == helpfulness.model_arm_order(
        "nonce-000"
    )
    assert tuple(field.name for field in fields(helpfulness.PreparedModelCall)) == (
        "prompt",
        "context_start",
        "context_end",
        "input_token_ids",
        "consumer_input_receipt",
        "expected_receipt",
        "rendered_context_token_count",
    )


def test_receiptless_case_attempts_all_six_arms_once_without_replacement(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    receiptless = {id(call.prepared_call) for call in manifest.cases[0].arm_calls}

    class SelectivelyReceiptlessBoundary:
        def __init__(self) -> None:
            self.attempted: list[int] = []

        def submit(
            self,
            input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            self.attempted.append(id(prepared_call))
            receipt = None
            if id(prepared_call) not in receiptless:
                receipt = helpfulness.make_model_call_receipt(
                    submitted_prompt=prepared_call.prompt,
                    context_start=prepared_call.context_start,
                    context_end=prepared_call.context_end,
                    input_token_ids=input_token_ids,
                )
            return helpfulness.ModelBoundaryOutput("UNKNOWN", receipt)

    boundary = SelectivelyReceiptlessBoundary()
    result = helpfulness.run_model_dry_run(
        manifest,
        _ByteModelTokenizer(),
        boundary,
    )

    assert result.validity == "invalid"
    assert tuple(
        (failure.case_index, failure.arm, failure.reason)
        for failure in result.invalid_calls
    ) == tuple(
        (0, arm, "model_call_receipt")
        for arm in helpfulness.model_arm_order("nonce-000")
    )
    case_zero = tuple(call for call in result.calls if call.case_index == 0)
    assert len(case_zero) == 6
    assert tuple(call.arm for call in case_zero) == helpfulness.model_arm_order(
        "nonce-000"
    )
    assert all(call.execution.valid is False for call in case_zero)
    assert all(
        call.execution.invalid_reason == "model_call_receipt" for call in case_zero
    )
    assert all(call.attempt_index == 0 for call in result.calls)
    assert len(boundary.attempted) == 384
    assert len(set(boundary.attempted)) == 384
    assert tuple(call.case_index for call in result.calls[::6]) == tuple(range(64))


def test_boundary_exception_keeps_fixed_slot_attrition_without_retry(
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    target_case_index = 7
    target_arm = "raw_history"
    target_call = next(
        call
        for call in manifest.cases[target_case_index].arm_calls
        if call.arm == target_arm
    )

    class OneFailingBoundary:
        def __init__(self) -> None:
            self.attempted: list[int] = []

        def submit(
            self,
            input_token_ids: tuple[int, ...],
            *,
            prepared_call: helpfulness.PreparedModelCall,
        ) -> helpfulness.ModelBoundaryOutput:
            self.attempted.append(id(prepared_call))
            if prepared_call is target_call.prepared_call:
                raise RuntimeError("fixed-slot failure")
            return helpfulness.ModelBoundaryOutput(
                response="UNKNOWN",
                receipt=helpfulness.make_model_call_receipt(
                    submitted_prompt=prepared_call.prompt,
                    context_start=prepared_call.context_start,
                    context_end=prepared_call.context_end,
                    input_token_ids=input_token_ids,
                ),
            )

    boundary = OneFailingBoundary()
    result = helpfulness.run_model_dry_run(
        manifest,
        _ByteModelTokenizer(),
        boundary,
    )

    assert result.validity == "invalid"
    assert result.invalid_calls == (
        helpfulness.ModelDryRunInvalidCall(
            case_index=target_case_index,
            arm=target_arm,
            attempted=True,
            reason="model_call_failure",
        ),
    )
    failed_slot = next(
        call
        for call in result.calls
        if call.case_index == target_case_index and call.arm == target_arm
    )
    assert failed_slot.attempt_index == 0
    assert failed_slot.execution is None
    assert len(result.calls) == 384
    assert len(boundary.attempted) == 384
    assert len(set(boundary.attempted)) == 384


def test_dry_run_does_not_swallow_evaluator_assertion(
    monkeypatch: pytest.MonkeyPatch,
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    boundary = _RecordingModelBoundary()

    def evaluator_bug(*_args, **_kwargs):
        raise AssertionError("evaluator invariant failed")

    monkeypatch.setattr(helpfulness, "submit_model_call", evaluator_bug)

    with pytest.raises(AssertionError, match="evaluator invariant failed"):
        helpfulness.run_model_dry_run(
            manifest,
            _ByteModelTokenizer(),
            boundary,
        )


def test_dry_run_does_not_swallow_unrelated_model_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
    prepared_model_manifest: helpfulness.ModelRunManifest,
) -> None:
    manifest = prepared_model_manifest
    boundary = _RecordingModelBoundary()

    def evaluator_protocol_bug(*_args, **_kwargs):
        raise helpfulness.ModelProtocolError("evaluator_protocol_bug")

    monkeypatch.setattr(helpfulness, "submit_model_call", evaluator_protocol_bug)

    with pytest.raises(helpfulness.ModelProtocolError) as error:
        helpfulness.run_model_dry_run(
            manifest,
            _ByteModelTokenizer(),
            boundary,
        )

    assert type(error.value) is helpfulness.ModelProtocolError
    assert error.value.reason == "evaluator_protocol_bug"


def test_model_preregistration_tokenizer_failure_is_not_candidate_rejection() -> None:
    class ExplodingTokenizer:
        def encode(
            self,
            _value: bytes,
            *,
            add_special_tokens: bool,
        ) -> tuple[int, ...]:
            del add_special_tokens
            raise RuntimeError("boom")

    result = helpfulness.prepare_model_case_registration(0, ExplodingTokenizer())

    assert result.registration is None
    assert result.failure == helpfulness.ModelManifestFailure(
        case_index=0,
        reason="tokenizer_failure",
        attempted_model_candidates=1,
    )


def _forbidden_direct_eager_imports(source: str) -> frozenset[str]:
    """Return forbidden imports executed eagerly while defining the module."""

    class EagerImportVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.paths: set[str] = set()
            self.names: set[str] = set()

        def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, _node: ast.Lambda) -> None:
            return

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                self.paths.add(alias.name)
                self.names.add(alias.name.rsplit(".", maxsplit=1)[-1])
                if alias.asname is not None:
                    self.names.add(alias.asname)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = f"{'.' * node.level}{node.module or ''}"
            for alias in node.names:
                separator = "" if not module or module.endswith(".") else "."
                self.paths.add(f"{module}{separator}{alias.name}")
                self.names.add(alias.name)
                if alias.asname is not None:
                    self.names.add(alias.asname)

    tree = ast.parse(source)
    visitor = EagerImportVisitor()
    visitor.visit(tree)

    bridge_module = "areal.v2.inference_service.inf_bridge"
    forbidden = {
        path
        for path in visitor.paths
        if path.split(".", maxsplit=1)[0] in {"torch", "transformers", "numpy"}
        or path == bridge_module
        or path.startswith(f"{bridge_module}.")
    }
    forbidden.update(
        f"name:{name}"
        for name in visitor.names
        if name in {"InfBridge", "ModelRequest", "AgentMetadata"}
    )
    return frozenset(forbidden)


@pytest.mark.parametrize(
    ("source", "expected_path"),
    [
        ("if True:\n    import torch\n", "torch"),
        (
            "try:\n"
            "    import areal.v2.inference_service.inf_bridge\n"
            "except ImportError:\n"
            "    pass\n",
            "areal.v2.inference_service.inf_bridge",
        ),
        (
            "from areal.v2.inference_service import inf_bridge\n",
            "areal.v2.inference_service.inf_bridge",
        ),
        ("from safe_module import InfBridge\n", "name:InfBridge"),
        ("import safe_module as ModelRequest\n", "name:ModelRequest"),
        (
            "from safe_module import metadata as AgentMetadata\n",
            "name:AgentMetadata",
        ),
    ],
)
def test_direct_eager_import_guard_catches_forbidden_forms(
    source: str,
    expected_path: str,
) -> None:
    assert expected_path in _forbidden_direct_eager_imports(source)


def test_direct_eager_import_guard_allows_function_local_numpy() -> None:
    source = "def lazy():\n    import numpy\n"

    assert _forbidden_direct_eager_imports(source) == frozenset()


def test_model_harness_adds_no_direct_eager_heavy_or_bridge_imports() -> None:
    source = Path(helpfulness.__file__).read_text()

    assert _forbidden_direct_eager_imports(source) == frozenset()
