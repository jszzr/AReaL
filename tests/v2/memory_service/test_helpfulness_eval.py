# SPDX-License-Identifier: Apache-2.0

"""Tests for the deterministic Memory Service helpfulness evaluator."""

from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import asdict
from hashlib import sha256

import pytest

from examples.memory_service import scoped_codebook_eval as helpfulness

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
