# SPDX-License-Identifier: Apache-2.0

"""Frozen causal evaluator for the local Memory update-policy experiment.

The evaluator separates the experiment into auditable data
boundaries:

1. freeze every case's candidate slots and the global category multiset, but
   not which category or key any case will be scored on;
2. bind all three update-policy decisions and releases;
3. seal consistent database snapshots;
4. accept one post-update beacon that assigns categories to cases; and only
5. use a domain-separated nonce to select one key inside each category.

This does not turn Python into a malicious-code sandbox.  It does make the
honest runner's causal ordering explicit and auditable: a policy cannot tune an
update to the future category or concrete key because neither is assigned
until after its decision and release are sealed.

The public run rows deliberately contain neither target keys, expected values,
nor raw responses.  They retain hashes, exact-match bits, and negative-control
labels, which is sufficient for paired accuracy and an exact sign test without
putting scorer answers in the canonical run root.  This is a no-plaintext
integrity property, not confidentiality: the small code space is enumerable.
The deterministic public suite is a contract/development benchmark, not a
hidden test for ranking arbitrary policies, and its fixed-corpus sign test is a
diagnostic rather than population-level evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "CASE_COUNT",
    "CASE_SEED",
    "CATEGORY_CASE_COUNTS",
    "CATEGORY_SLOT_COUNTS",
    "ENTRY_CATEGORIES",
    "ADVERSARIAL_STRESS_CATEGORIES",
    "PLACEBO_CONTROL_CATEGORIES",
    "POLICIES",
    "POLICY_SCHEDULE_ALGORITHM",
    "CaseDecisionSealV1",
    "CausalEvalMetricsV1",
    "CausalEvalRowV1",
    "CategoryAccuracyV1",
    "FrozenCaseV1",
    "FrozenEvidenceV1",
    "FrozenSlotV1",
    "LocalUpdateCausalEvalError",
    "PairedComparisonV1",
    "PolicyAccuracyV1",
    "PolicyDecisionCommitmentV1",
    "PolicyOutcomeV1",
    "SlottedCaseV1",
    "TargetCategoryScheduleV1",
    "analyze_causal_eval_v1",
    "no_plaintext_run_wire_v1",
    "causal_eval_root_sha256_v1",
    "frozen_case_sha256_v1",
    "frozen_suite_sha256_v1",
    "generate_frozen_cases_v1",
    "make_case_decision_seal_v1",
    "make_causal_eval_row_v1",
    "make_target_category_schedule_v1",
    "policy_order_v1",
    "reference_policy_response_v1",
    "slot_frozen_case_v1",
    "validate_complete_causal_eval_v1",
]


SCHEMA_VERSION = 1
CASE_COUNT = 64
CASE_SEED = "areal-memory-local-update-causal-v1-20260712"
POLICIES = ("feedback_latest", "noop", "latest_any")
ENTRY_CATEGORIES = (
    "corrected",
    "paraphrased_corrected",
    "stable_placebo",
    "corrupt_feedback",
    # A stable base plus feedback that contains no new fact at all.
    "no_fact_feedback",
)
PLACEBO_CONTROL_CATEGORIES = ("stable_placebo", "no_fact_feedback")
ADVERSARIAL_STRESS_CATEGORIES = ("corrupt_feedback",)
CATEGORY_SLOT_COUNTS = (
    ("corrected", 3),
    ("paraphrased_corrected", 1),
    ("stable_placebo", 2),
    ("corrupt_feedback", 1),
    ("no_fact_feedback", 1),
)
CATEGORY_CASE_COUNTS = (
    ("corrected", 24),
    ("paraphrased_corrected", 8),
    ("stable_placebo", 16),
    ("corrupt_feedback", 8),
    ("no_fact_feedback", 8),
)
POLICY_SCHEDULE_ALGORITHM = "logical-williams-3-policy-2-block-v1"

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_KEY_ALPHABET = "abcdefghjklmnpqrstuvwxyz23456789"
_KEY_PATTERN = re.compile(r"project-[abcdefghjklmnpqrstuvwxyz23456789]{6}")
_VALUE_PATTERN = re.compile(r"[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CASE_ID_PATTERN = re.compile(r"lcase_[0-9a-f]{24}")
_FACT_PATTERN = re.compile(
    r"(?P<key>project-[abcdefghjklmnpqrstuvwxyz23456789]{6}) = "
    r"(?P<value>[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5})"
)
_FROZEN_CASE_DOMAIN = b"areal-memory-local-update-frozen-case-v1\0"
_FROZEN_SUITE_DOMAIN = b"areal-memory-local-update-frozen-suite-v1\0"
_DECISION_SEAL_DOMAIN = b"areal-memory-local-update-decision-seal-v1\0"
_TARGET_CATEGORY_DOMAIN = b"areal-memory-local-update-target-category-v1\0"
_TARGET_CATEGORY_SCHEDULE_DOMAIN = (
    b"areal-memory-local-update-target-category-schedule-v1\0"
)
_FUTURE_SLOT_DOMAIN = b"areal-memory-local-update-future-slot-v1\0"
_SLOTTED_CASE_DOMAIN = b"areal-memory-local-update-slotted-case-v1\0"
_RUN_ROOT_DOMAIN = b"areal-memory-local-update-causal-run-v1\0"
_FROZEN_SUITE_SHA256 = (
    "4f096544014d7ef46006eef5359de38b6a4ab4a9a698971e1d02a8d092b4be1d"
)

# A Williams design for three treatments needs two mirrored 3-row blocks.
# Across a complete block this balances parent-side presentation.  The runner
# randomizes actual child wire order by opaque token, so these rows make no
# process-order or carryover-control claim.
_POLICY_SCHEDULE_ROWS = (
    ("feedback_latest", "noop", "latest_any"),
    ("noop", "latest_any", "feedback_latest"),
    ("latest_any", "feedback_latest", "noop"),
    ("latest_any", "noop", "feedback_latest"),
    ("feedback_latest", "latest_any", "noop"),
    ("noop", "feedback_latest", "latest_any"),
)


class LocalUpdateCausalEvalError(ValueError):
    """Stable failure reason that does not echo scorer plaintext."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("causal-eval reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class FrozenEvidenceV1:
    """One pre-seal evidence event, expressed without storage-specific IDs."""

    kind: str
    payload: str
    sequence_no: int
    observed_offset_seconds: int


@dataclass(frozen=True, slots=True)
class FrozenSlotV1:
    """One codebook fact and the two evidence events that can update it."""

    slot: int
    category: str
    key: str
    base_value: str
    expected_value: str
    evidence: tuple[FrozenEvidenceV1, ...]


@dataclass(frozen=True, slots=True)
class FrozenCaseV1:
    """Private case manifest frozen before any update policy is invoked."""

    schema_version: int
    seed: str
    case_index: int
    case_id: str
    slots: tuple[FrozenSlotV1, ...]


@dataclass(frozen=True, slots=True)
class PolicyDecisionCommitmentV1:
    policy: str
    decision_sha256: str


@dataclass(frozen=True, slots=True)
class CaseDecisionSealV1:
    """Opaque policy commitments made before the future nonce is accepted.

    The integration runner, outside this pure module, must prove that each hash
    came from its validated policy input, decision, application, and release.
    """

    schema_version: int
    case_index: int
    frozen_case_sha256: str
    policy_decisions: tuple[PolicyDecisionCommitmentV1, ...]
    seal_sha256: str


@dataclass(frozen=True, slots=True)
class TargetCategoryScheduleV1:
    """Post-update category assignment derived from one run-level nonce."""

    schema_version: int
    category_nonce: str
    categories: tuple[str, ...]
    schedule_sha256: str


@dataclass(frozen=True, slots=True)
class SlottedCaseV1:
    """Private scorer view created from a nonce after the decision seal."""

    schema_version: int
    case_index: int
    case_id: str
    frozen_case_sha256: str
    decision_seal_sha256: str
    target_category_schedule_sha256: str
    future_nonce: str
    target_category: str
    target_slot: int
    target_key: str
    expected_value: str
    query: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class PolicyOutcomeV1:
    """Answer-free receipt for one policy response."""

    policy: str
    response_sha256: str
    response_utf8_bytes: int
    exact_match: bool


@dataclass(frozen=True, slots=True)
class CausalEvalRowV1:
    """One complete paired subject; raw queries and answers are absent."""

    schema_version: int
    case_index: int
    case_id: str
    frozen_case_sha256: str
    decision_seal_sha256: str
    target_category_schedule_sha256: str
    slotted_case_sha256: str
    target_category: str
    policy_schedule: tuple[str, ...]
    outcomes: tuple[PolicyOutcomeV1, ...]


@dataclass(frozen=True, slots=True)
class PolicyAccuracyV1:
    policy: str
    successes: int
    total: int
    accuracy: float


@dataclass(frozen=True, slots=True)
class CategoryAccuracyV1:
    policy: str
    category: str
    successes: int
    total: int
    accuracy: float
    control_role: str


@dataclass(frozen=True, slots=True)
class PairedComparisonV1:
    left_policy: str
    right_policy: str
    wins: int
    losses: int
    ties: int
    mean_delta: float
    macro_category_delta: float
    discordant_count: int
    exact_sign_test_p_value: float


@dataclass(frozen=True, slots=True)
class CausalEvalMetricsV1:
    schema_version: int
    case_count: int
    no_plaintext_root_sha256: str
    policy_accuracies: tuple[PolicyAccuracyV1, ...]
    category_accuracies: tuple[CategoryAccuracyV1, ...]
    paired_comparisons: tuple[PairedComparisonV1, ...]
    aggregate_weighting: str
    placebo_control_categories: tuple[str, ...]
    adversarial_stress_categories: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise LocalUpdateCausalEvalError("closed_schema") from error


def _token(domain: str, ordinal: int, attempt: int, *, alphabet: str, size: int) -> str:
    material = f"{CASE_SEED}|{domain}|{ordinal}|{attempt}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    return "".join(alphabet[byte % len(alphabet)] for byte in digest[:size])


def _unique_token(
    domain: str,
    ordinal: int,
    *,
    alphabet: str,
    size: int,
    used: set[str],
) -> str:
    for attempt in range(1_000_000):
        value = _token(domain, ordinal, attempt, alphabet=alphabet, size=size)
        if value not in used:
            used.add(value)
            return value
    raise AssertionError("deterministic token space unexpectedly exhausted")


def _shuffled_labels(
    counts: tuple[tuple[str, int], ...],
    *,
    domain: str,
) -> tuple[str, ...]:
    labelled = [
        (label, occurrence) for label, count in counts for occurrence in range(count)
    ]
    labelled.sort(
        key=lambda item: hashlib.sha256(
            f"{CASE_SEED}|{domain}|{item[0]}|{item[1]}".encode("ascii")
        ).digest()
    )
    return tuple(label for label, _occurrence in labelled)


def _frozen_evidence_wire(value: FrozenEvidenceV1) -> dict[str, object]:
    return {
        "kind": value.kind,
        "observed_offset_seconds": value.observed_offset_seconds,
        "payload": value.payload,
        "sequence_no": value.sequence_no,
    }


def _frozen_slot_wire(value: FrozenSlotV1) -> dict[str, object]:
    return {
        "base_value": value.base_value,
        "category": value.category,
        "evidence": [_frozen_evidence_wire(item) for item in value.evidence],
        "expected_value": value.expected_value,
        "key": value.key,
        "slot": value.slot,
    }


def _frozen_case_body(value: FrozenCaseV1) -> dict[str, object]:
    return {
        "case_index": value.case_index,
        "schema_version": value.schema_version,
        "seed": value.seed,
        "slots": [_frozen_slot_wire(item) for item in value.slots],
    }


def frozen_case_sha256_v1(value: FrozenCaseV1) -> str:
    if type(value) is not FrozenCaseV1:
        raise LocalUpdateCausalEvalError("closed_schema")
    try:
        body = _frozen_case_body(value)
    except AttributeError as error:
        raise LocalUpdateCausalEvalError("closed_schema") from error
    return hashlib.sha256(_FROZEN_CASE_DOMAIN + _canonical_json_bytes(body)).hexdigest()


def _make_evidence(
    *,
    slot: int,
    category: str,
    key: str,
    base_value: str,
    expected_value: str,
    first_noise: str,
    second_noise: str,
) -> tuple[FrozenEvidenceV1, ...]:
    if category == "corrected":
        feedback_payload = f"{key} = {expected_value}"
        later_payload = f"{key} = {second_noise}"
    elif category == "paraphrased_corrected":
        templates = (
            "Correction noted: {key} should now be {value}.",
            "The current code for {key} has changed to {value}.",
            "Use {value} as the updated code for {key}.",
        )
        feedback_payload = templates[slot % len(templates)].format(
            key=key,
            value=expected_value,
        )
        later_payload = f"{key} = {second_noise}"
    elif category == "stable_placebo":
        feedback_payload = f"{key} = {base_value}"
        later_payload = f"{key} = {base_value}"
    elif category == "corrupt_feedback":
        feedback_payload = f"{key} = {first_noise}"
        later_payload = f"{key} = {first_noise}"
    elif category == "no_fact_feedback":
        # A genuine no-fact negative control: there is no proposed value to
        # extract, unlike the valid paraphrased correction above.
        feedback_payload = f"Please verify whether {key} needs an update."
        later_payload = f"{key} = {second_noise}"
    else:
        raise AssertionError("unknown frozen category")
    return (
        FrozenEvidenceV1(
            kind="feedback",
            payload=feedback_payload,
            sequence_no=slot,
            observed_offset_seconds=slot,
        ),
        FrozenEvidenceV1(
            kind="user_message",
            payload=later_payload,
            sequence_no=8 + slot,
            observed_offset_seconds=8 + slot,
        ),
    )


def _build_frozen_cases_v1() -> tuple[FrozenCaseV1, ...]:
    used_keys: set[str] = set()
    used_values: set[str] = set()
    cases: list[FrozenCaseV1] = []
    ordinal = 0
    for case_index in range(CASE_COUNT):
        layout = _shuffled_labels(
            CATEGORY_SLOT_COUNTS,
            domain=f"slot-categories-{case_index}",
        )
        slots: list[FrozenSlotV1] = []
        for slot, category in enumerate(layout):
            key = "project-" + _unique_token(
                "key",
                ordinal,
                alphabet=_KEY_ALPHABET,
                size=6,
                used=used_keys,
            )
            base_value = _unique_token(
                "value-base",
                ordinal,
                alphabet=_ALPHABET,
                size=5,
                used=used_values,
            )
            corrected_value = _unique_token(
                "value-corrected",
                ordinal,
                alphabet=_ALPHABET,
                size=5,
                used=used_values,
            )
            first_noise = _unique_token(
                "value-first-noise",
                ordinal,
                alphabet=_ALPHABET,
                size=5,
                used=used_values,
            )
            second_noise = _unique_token(
                "value-second-noise",
                ordinal,
                alphabet=_ALPHABET,
                size=5,
                used=used_values,
            )
            expected_value = (
                corrected_value
                if category in {"corrected", "paraphrased_corrected"}
                else base_value
            )
            evidence = _make_evidence(
                slot=slot,
                category=category,
                key=key,
                base_value=base_value,
                expected_value=expected_value,
                first_noise=first_noise,
                second_noise=second_noise,
            )
            slots.append(
                FrozenSlotV1(
                    slot=slot,
                    category=category,
                    key=key,
                    base_value=base_value,
                    expected_value=expected_value,
                    evidence=evidence,
                )
            )
            ordinal += 1
        provisional = FrozenCaseV1(
            schema_version=SCHEMA_VERSION,
            seed=CASE_SEED,
            case_index=case_index,
            case_id="lcase_" + "0" * 24,
            slots=tuple(slots),
        )
        digest = frozen_case_sha256_v1(provisional)
        cases.append(
            FrozenCaseV1(
                schema_version=provisional.schema_version,
                seed=provisional.seed,
                case_index=provisional.case_index,
                case_id=f"lcase_{digest[:24]}",
                slots=provisional.slots,
            )
        )
    return tuple(cases)


def frozen_suite_sha256_v1(cases: tuple[FrozenCaseV1, ...]) -> str:
    if type(cases) is not tuple:
        raise LocalUpdateCausalEvalError("closed_schema")
    value = {
        "case_count": len(cases),
        "case_sha256": [frozen_case_sha256_v1(case) for case in cases],
        "schema_version": SCHEMA_VERSION,
    }
    return hashlib.sha256(
        _FROZEN_SUITE_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _validate_frozen_cases_structure(cases: tuple[FrozenCaseV1, ...]) -> None:
    if (
        type(cases) is not tuple
        or len(cases) != CASE_COUNT
        or any(type(case) is not FrozenCaseV1 for case in cases)
        or tuple(case.case_index for case in cases) != tuple(range(CASE_COUNT))
    ):
        raise LocalUpdateCausalEvalError("frozen_suite")
    all_keys: set[str] = set()
    value_owners: dict[str, tuple[int, int]] = {}
    for case in cases:
        if (
            type(case.schema_version) is not int
            or case.schema_version != SCHEMA_VERSION
            or type(case.seed) is not str
            or case.seed != CASE_SEED
            or type(case.case_index) is not int
            or type(case.case_id) is not str
            or _CASE_ID_PATTERN.fullmatch(case.case_id) is None
            or case.case_id != f"lcase_{frozen_case_sha256_v1(case)[:24]}"
            or type(case.slots) is not tuple
            or len(case.slots) != 8
            or any(type(slot) is not FrozenSlotV1 for slot in case.slots)
            or tuple(slot.slot for slot in case.slots) != tuple(range(8))
        ):
            raise LocalUpdateCausalEvalError("frozen_case")
        slot_counts = {category: 0 for category in ENTRY_CATEGORIES}
        case_sequences: set[int] = set()
        for slot in case.slots:
            if (
                type(slot.slot) is not int
                or type(slot.category) is not str
                or slot.category not in ENTRY_CATEGORIES
                or type(slot.key) is not str
                or _KEY_PATTERN.fullmatch(slot.key) is None
                or type(slot.base_value) is not str
                or _VALUE_PATTERN.fullmatch(slot.base_value) is None
                or type(slot.expected_value) is not str
                or _VALUE_PATTERN.fullmatch(slot.expected_value) is None
                or type(slot.evidence) is not tuple
                or len(slot.evidence) != 2
                or any(type(item) is not FrozenEvidenceV1 for item in slot.evidence)
            ):
                raise LocalUpdateCausalEvalError("frozen_slot")
            if slot.key in all_keys:
                raise LocalUpdateCausalEvalError("token_collision")
            all_keys.add(slot.key)
            slot_counts[slot.category] += 1
            if slot.category in {"corrected", "paraphrased_corrected"}:
                if slot.expected_value == slot.base_value:
                    raise LocalUpdateCausalEvalError("frozen_semantics")
            elif slot.expected_value != slot.base_value:
                raise LocalUpdateCausalEvalError("frozen_semantics")
            local_values = {slot.base_value, slot.expected_value}
            for event in slot.evidence:
                if (
                    type(event.kind) is not str
                    or event.kind not in ("feedback", "user_message")
                    or type(event.payload) is not str
                    or type(event.sequence_no) is not int
                    or event.sequence_no < 0
                    or type(event.observed_offset_seconds) is not int
                    or event.observed_offset_seconds != event.sequence_no
                    or event.sequence_no in case_sequences
                ):
                    raise LocalUpdateCausalEvalError("frozen_evidence")
                case_sequences.add(event.sequence_no)
                local_values.update(_VALUE_PATTERN.findall(event.payload))
            first_match = _FACT_PATTERN.fullmatch(slot.evidence[0].payload)
            first_tokens = _VALUE_PATTERN.findall(slot.evidence[0].payload)
            second_match = _FACT_PATTERN.fullmatch(slot.evidence[1].payload)
            if second_match is None:
                raise LocalUpdateCausalEvalError("frozen_semantics")
            if slot.category in {
                "corrected",
                "stable_placebo",
                "corrupt_feedback",
            }:
                first_valid = first_match is not None
            elif slot.category == "paraphrased_corrected":
                first_valid = (
                    first_match is None
                    and len(first_tokens) == 1
                    and first_tokens[0] == slot.expected_value
                    and slot.key in slot.evidence[0].payload
                )
            else:
                first_valid = first_match is None and not first_tokens
            if not first_valid:
                raise LocalUpdateCausalEvalError("frozen_semantics")
            expected_evidence = _make_evidence(
                slot=slot.slot,
                category=slot.category,
                key=slot.key,
                base_value=slot.base_value,
                expected_value=slot.expected_value,
                first_noise=(
                    first_match.group("value")
                    if first_match is not None
                    else (
                        first_tokens[0] if first_tokens else second_match.group("value")
                    )
                ),
                second_noise=second_match.group("value"),
            )
            if slot.evidence != expected_evidence:
                raise LocalUpdateCausalEvalError("frozen_semantics")
            owner = (case.case_index, slot.slot)
            for token in local_values:
                previous = value_owners.setdefault(token, owner)
                if previous != owner:
                    raise LocalUpdateCausalEvalError("token_collision")
        if tuple(sorted(slot_counts.items())) != tuple(
            sorted(CATEGORY_SLOT_COUNTS)
        ) or case_sequences != set(range(16)):
            raise LocalUpdateCausalEvalError("frozen_case")


@lru_cache(maxsize=1)
def generate_frozen_cases_v1() -> tuple[FrozenCaseV1, ...]:
    """Return the immutable 64-case noisy-codebook suite."""

    cases = _build_frozen_cases_v1()
    _validate_frozen_cases_structure(cases)
    digest = frozen_suite_sha256_v1(cases)
    if digest != _FROZEN_SUITE_SHA256:
        raise AssertionError("frozen local-update suite drifted")
    return cases


def _derive_target_categories(
    cases: tuple[FrozenCaseV1, ...],
    category_nonce: str,
) -> tuple[str, ...]:
    ranked_cases = sorted(
        cases,
        key=lambda case: (
            hashlib.sha256(
                _TARGET_CATEGORY_DOMAIN
                + bytes.fromhex(category_nonce)
                + case.case_id.encode("ascii")
            ).digest(),
            case.case_id,
        ),
    )
    labels = tuple(
        category
        for category, count in CATEGORY_CASE_COUNTS
        for _occurrence in range(count)
    )
    if len(labels) != CASE_COUNT:
        raise AssertionError("target-category multiset must contain 64 labels")
    category_by_index = {
        case.case_index: category
        for case, category in zip(ranked_cases, labels, strict=True)
    }
    return tuple(category_by_index[index] for index in range(CASE_COUNT))


def _target_category_schedule_body(
    cases: tuple[FrozenCaseV1, ...],
    value: TargetCategoryScheduleV1,
) -> dict[str, object]:
    return {
        "categories": list(value.categories),
        "category_nonce": value.category_nonce,
        "frozen_suite_sha256": frozen_suite_sha256_v1(cases),
        "schema_version": value.schema_version,
    }


def _validate_target_category_schedule(
    cases: tuple[FrozenCaseV1, ...],
    value: TargetCategoryScheduleV1,
) -> None:
    if (
        type(value) is not TargetCategoryScheduleV1
        or type(value.schema_version) is not int
        or value.schema_version != SCHEMA_VERSION
        or type(value.category_nonce) is not str
        or _SHA256_PATTERN.fullmatch(value.category_nonce) is None
        or type(value.categories) is not tuple
        or len(value.categories) != CASE_COUNT
        or value.categories != _derive_target_categories(cases, value.category_nonce)
        or type(value.schedule_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.schedule_sha256) is None
    ):
        raise LocalUpdateCausalEvalError("target_category_schedule")
    expected = hashlib.sha256(
        _TARGET_CATEGORY_SCHEDULE_DOMAIN
        + _canonical_json_bytes(_target_category_schedule_body(cases, value))
    ).hexdigest()
    if value.schedule_sha256 != expected:
        raise LocalUpdateCausalEvalError("target_category_schedule")


def make_target_category_schedule_v1(
    cases: tuple[FrozenCaseV1, ...],
    *,
    category_nonce: str,
) -> TargetCategoryScheduleV1:
    """Assign the preregistered category multiset only after update sealing."""

    _validate_frozen_cases_structure(cases)
    if cases != generate_frozen_cases_v1():
        raise LocalUpdateCausalEvalError("frozen_suite")
    if (
        type(category_nonce) is not str
        or _SHA256_PATTERN.fullmatch(category_nonce) is None
    ):
        raise LocalUpdateCausalEvalError("target_category_schedule")
    provisional = TargetCategoryScheduleV1(
        schema_version=SCHEMA_VERSION,
        category_nonce=category_nonce,
        categories=_derive_target_categories(cases, category_nonce),
        schedule_sha256="0" * 64,
    )
    result = TargetCategoryScheduleV1(
        schema_version=provisional.schema_version,
        category_nonce=provisional.category_nonce,
        categories=provisional.categories,
        schedule_sha256=hashlib.sha256(
            _TARGET_CATEGORY_SCHEDULE_DOMAIN
            + _canonical_json_bytes(_target_category_schedule_body(cases, provisional))
        ).hexdigest(),
    )
    _validate_target_category_schedule(cases, result)
    return result


def _validate_registered_case(case: FrozenCaseV1) -> None:
    if (
        type(case) is not FrozenCaseV1
        or type(case.case_index) is not int
        or case.case_index not in range(CASE_COUNT)
        or case != generate_frozen_cases_v1()[case.case_index]
    ):
        raise LocalUpdateCausalEvalError("frozen_case")


def policy_order_v1(case_index: int) -> tuple[str, ...]:
    if type(case_index) is not int or case_index not in range(CASE_COUNT):
        raise LocalUpdateCausalEvalError("case_index")
    return _POLICY_SCHEDULE_ROWS[case_index % len(_POLICY_SCHEDULE_ROWS)]


def _decision_seal_body(value: CaseDecisionSealV1) -> dict[str, object]:
    return {
        "case_index": value.case_index,
        "frozen_case_sha256": value.frozen_case_sha256,
        "policy_decisions": [
            {
                "decision_sha256": item.decision_sha256,
                "policy": item.policy,
            }
            for item in value.policy_decisions
        ],
        "schema_version": value.schema_version,
    }


def _validate_case_decision_seal(
    case: FrozenCaseV1,
    seal: CaseDecisionSealV1,
) -> None:
    _validate_registered_case(case)
    if (
        type(seal) is not CaseDecisionSealV1
        or type(seal.schema_version) is not int
        or seal.schema_version != SCHEMA_VERSION
        or type(seal.case_index) is not int
        or seal.case_index != case.case_index
        or type(seal.frozen_case_sha256) is not str
        or seal.frozen_case_sha256 != frozen_case_sha256_v1(case)
        or type(seal.policy_decisions) is not tuple
        or any(
            type(item) is not PolicyDecisionCommitmentV1
            or type(item.policy) is not str
            or type(item.decision_sha256) is not str
            or _SHA256_PATTERN.fullmatch(item.decision_sha256) is None
            for item in seal.policy_decisions
        )
        or tuple(item.policy for item in seal.policy_decisions) != POLICIES
        or type(seal.seal_sha256) is not str
    ):
        raise LocalUpdateCausalEvalError("decision_seal")
    expected = hashlib.sha256(
        _DECISION_SEAL_DOMAIN + _canonical_json_bytes(_decision_seal_body(seal))
    ).hexdigest()
    if seal.seal_sha256 != expected:
        raise LocalUpdateCausalEvalError("decision_seal")


def make_case_decision_seal_v1(
    case: FrozenCaseV1,
    decision_sha256_by_policy: Mapping[str, str],
) -> CaseDecisionSealV1:
    """Commit all three policy decisions before accepting a future nonce."""

    _validate_registered_case(case)
    if not isinstance(decision_sha256_by_policy, Mapping):
        raise LocalUpdateCausalEvalError("closed_schema")
    if set(decision_sha256_by_policy) != set(POLICIES) or any(
        type(key) is not str
        or type(value) is not str
        or _SHA256_PATTERN.fullmatch(value) is None
        for key, value in decision_sha256_by_policy.items()
    ):
        raise LocalUpdateCausalEvalError("decision_seal")
    commitments = tuple(
        PolicyDecisionCommitmentV1(
            policy=policy,
            decision_sha256=decision_sha256_by_policy[policy],
        )
        for policy in POLICIES
    )
    provisional = CaseDecisionSealV1(
        schema_version=SCHEMA_VERSION,
        case_index=case.case_index,
        frozen_case_sha256=frozen_case_sha256_v1(case),
        policy_decisions=commitments,
        seal_sha256="0" * 64,
    )
    seal = CaseDecisionSealV1(
        schema_version=provisional.schema_version,
        case_index=provisional.case_index,
        frozen_case_sha256=provisional.frozen_case_sha256,
        policy_decisions=provisional.policy_decisions,
        seal_sha256=hashlib.sha256(
            _DECISION_SEAL_DOMAIN
            + _canonical_json_bytes(_decision_seal_body(provisional))
        ).hexdigest(),
    )
    _validate_case_decision_seal(case, seal)
    return seal


def _slotted_case_body(value: SlottedCaseV1) -> dict[str, object]:
    return {
        "case_id": value.case_id,
        "case_index": value.case_index,
        "decision_seal_sha256": value.decision_seal_sha256,
        "expected_value": value.expected_value,
        "frozen_case_sha256": value.frozen_case_sha256,
        "future_nonce": value.future_nonce,
        "query": value.query,
        "schema_version": value.schema_version,
        "target_category": value.target_category,
        "target_category_schedule_sha256": (value.target_category_schedule_sha256),
        "target_key": value.target_key,
        "target_slot": value.target_slot,
    }


def _expected_target_slot(
    case: FrozenCaseV1,
    seal: CaseDecisionSealV1,
    target_category: str,
    future_nonce: str,
) -> int:
    eligible = tuple(
        slot.slot for slot in case.slots if slot.category == target_category
    )
    if not eligible:
        raise LocalUpdateCausalEvalError("frozen_case")
    return min(
        eligible,
        key=lambda slot: (
            hashlib.sha256(
                _FUTURE_SLOT_DOMAIN
                + bytes.fromhex(seal.seal_sha256)
                + bytes.fromhex(future_nonce)
                + target_category.encode("ascii")
                + slot.to_bytes(2, "big")
            ).digest(),
            slot,
        ),
    )


def _validate_slotted_case(
    case: FrozenCaseV1,
    seal: CaseDecisionSealV1,
    target_schedule: TargetCategoryScheduleV1,
    slotted: SlottedCaseV1,
) -> None:
    _validate_case_decision_seal(case, seal)
    cases = generate_frozen_cases_v1()
    _validate_target_category_schedule(cases, target_schedule)
    target_category = target_schedule.categories[case.case_index]
    if (
        type(slotted) is not SlottedCaseV1
        or type(slotted.schema_version) is not int
        or slotted.schema_version != SCHEMA_VERSION
        or type(slotted.case_index) is not int
        or slotted.case_index != case.case_index
        or type(slotted.case_id) is not str
        or slotted.case_id != case.case_id
        or type(slotted.frozen_case_sha256) is not str
        or slotted.frozen_case_sha256 != frozen_case_sha256_v1(case)
        or type(slotted.decision_seal_sha256) is not str
        or slotted.decision_seal_sha256 != seal.seal_sha256
        or type(slotted.target_category_schedule_sha256) is not str
        or slotted.target_category_schedule_sha256 != target_schedule.schedule_sha256
        or type(slotted.future_nonce) is not str
        or _SHA256_PATTERN.fullmatch(slotted.future_nonce) is None
        or type(slotted.target_category) is not str
        or slotted.target_category != target_category
        or type(slotted.target_slot) is not int
        or slotted.target_slot
        != _expected_target_slot(
            case,
            seal,
            target_category,
            slotted.future_nonce,
        )
        or type(slotted.target_key) is not str
        or type(slotted.expected_value) is not str
        or type(slotted.query) is not str
        or type(slotted.content_sha256) is not str
    ):
        raise LocalUpdateCausalEvalError("slotted_case")
    target = case.slots[slotted.target_slot]
    if (
        target.category != slotted.target_category
        or target.key != slotted.target_key
        or target.expected_value != slotted.expected_value
        or slotted.query
        != (
            f"What is the current code for {target.key}? "
            "Reply with exactly the code or UNKNOWN."
        )
    ):
        raise LocalUpdateCausalEvalError("slotted_case")
    expected_hash = hashlib.sha256(
        _SLOTTED_CASE_DOMAIN + _canonical_json_bytes(_slotted_case_body(slotted))
    ).hexdigest()
    if slotted.content_sha256 != expected_hash:
        raise LocalUpdateCausalEvalError("slotted_case")


def slot_frozen_case_v1(
    case: FrozenCaseV1,
    seal: CaseDecisionSealV1,
    target_schedule: TargetCategoryScheduleV1,
    *,
    future_nonce: str,
) -> SlottedCaseV1:
    """Select one concrete same-category key after all decisions are sealed.

    The pure function validates derivation, not randomness provenance.  A real
    run must derive each per-case nonce from one unpredictable run-level beacon
    obtained only after all 64-by-3 releases have been sealed; accepting a
    caller-chosen nonce per case would permit target-slot grinding.
    """

    if type(future_nonce) is not str or _SHA256_PATTERN.fullmatch(future_nonce) is None:
        raise LocalUpdateCausalEvalError("future_nonce")
    _validate_case_decision_seal(case, seal)
    cases = generate_frozen_cases_v1()
    _validate_target_category_schedule(cases, target_schedule)
    target_category = target_schedule.categories[case.case_index]
    target_slot = _expected_target_slot(
        case,
        seal,
        target_category,
        future_nonce,
    )
    target = case.slots[target_slot]
    provisional = SlottedCaseV1(
        schema_version=SCHEMA_VERSION,
        case_index=case.case_index,
        case_id=case.case_id,
        frozen_case_sha256=frozen_case_sha256_v1(case),
        decision_seal_sha256=seal.seal_sha256,
        target_category_schedule_sha256=target_schedule.schedule_sha256,
        future_nonce=future_nonce,
        target_category=target_category,
        target_slot=target.slot,
        target_key=target.key,
        expected_value=target.expected_value,
        query=(
            f"What is the current code for {target.key}? "
            "Reply with exactly the code or UNKNOWN."
        ),
        content_sha256="0" * 64,
    )
    slotted = SlottedCaseV1(
        schema_version=provisional.schema_version,
        case_index=provisional.case_index,
        case_id=provisional.case_id,
        frozen_case_sha256=provisional.frozen_case_sha256,
        decision_seal_sha256=provisional.decision_seal_sha256,
        target_category_schedule_sha256=(provisional.target_category_schedule_sha256),
        future_nonce=provisional.future_nonce,
        target_category=provisional.target_category,
        target_slot=provisional.target_slot,
        target_key=provisional.target_key,
        expected_value=provisional.expected_value,
        query=provisional.query,
        content_sha256=hashlib.sha256(
            _SLOTTED_CASE_DOMAIN
            + _canonical_json_bytes(_slotted_case_body(provisional))
        ).hexdigest(),
    )
    _validate_slotted_case(case, seal, target_schedule, slotted)
    return slotted


def _parse_fact(payload: str) -> tuple[str, str] | None:
    match = _FACT_PATTERN.fullmatch(payload)
    if match is None:
        return None
    return match.group("key"), match.group("value")


def reference_policy_response_v1(
    case: FrozenCaseV1,
    seal: CaseDecisionSealV1,
    target_schedule: TargetCategoryScheduleV1,
    slotted: SlottedCaseV1,
    policy: str,
) -> str:
    """Pure reference semantics used to verify the frozen experiment design."""

    _validate_registered_case(case)
    _validate_slotted_case(case, seal, target_schedule, slotted)
    if type(policy) is not str or policy not in POLICIES:
        raise LocalUpdateCausalEvalError("policy")
    values = {slot.key: slot.base_value for slot in case.slots}
    if policy != "noop":
        evidence = sorted(
            (item for slot in case.slots for item in slot.evidence),
            key=lambda item: (item.observed_offset_seconds, item.sequence_no),
        )
        for item in evidence:
            if policy == "feedback_latest" and item.kind != "feedback":
                continue
            fact = _parse_fact(item.payload)
            if fact is not None:
                values[fact[0]] = fact[1]
    if slotted.target_key not in values:
        raise LocalUpdateCausalEvalError("slotted_case")
    return values[slotted.target_key]


def _no_plaintext_row_value(row: CausalEvalRowV1) -> dict[str, object]:
    return {
        "case_id": row.case_id,
        "case_index": row.case_index,
        "decision_seal_sha256": row.decision_seal_sha256,
        "frozen_case_sha256": row.frozen_case_sha256,
        "outcomes": [
            {
                "exact_match": outcome.exact_match,
                "policy": outcome.policy,
                "response_sha256": outcome.response_sha256,
                "response_utf8_bytes": outcome.response_utf8_bytes,
            }
            for outcome in row.outcomes
        ],
        "policy_schedule": list(row.policy_schedule),
        "schema_version": row.schema_version,
        "slotted_case_sha256": row.slotted_case_sha256,
        "target_category": row.target_category,
        "target_category_schedule_sha256": (row.target_category_schedule_sha256),
    }


def _validate_row(
    case: FrozenCaseV1,
    seal: CaseDecisionSealV1,
    target_schedule: TargetCategoryScheduleV1,
    slotted: SlottedCaseV1,
    row: CausalEvalRowV1,
) -> None:
    _validate_slotted_case(case, seal, target_schedule, slotted)
    if (
        type(row) is not CausalEvalRowV1
        or type(row.schema_version) is not int
        or row.schema_version != SCHEMA_VERSION
        or type(row.case_index) is not int
        or row.case_index != case.case_index
        or type(row.case_id) is not str
        or row.case_id != case.case_id
        or type(row.frozen_case_sha256) is not str
        or row.frozen_case_sha256 != frozen_case_sha256_v1(case)
        or type(row.decision_seal_sha256) is not str
        or row.decision_seal_sha256 != seal.seal_sha256
        or type(row.target_category_schedule_sha256) is not str
        or row.target_category_schedule_sha256 != target_schedule.schedule_sha256
        or type(row.slotted_case_sha256) is not str
        or row.slotted_case_sha256 != slotted.content_sha256
        or type(row.target_category) is not str
        or row.target_category != slotted.target_category
        or type(row.policy_schedule) is not tuple
        or row.policy_schedule != policy_order_v1(case.case_index)
        or type(row.outcomes) is not tuple
        or any(type(outcome) is not PolicyOutcomeV1 for outcome in row.outcomes)
        or tuple(outcome.policy for outcome in row.outcomes) != row.policy_schedule
    ):
        raise LocalUpdateCausalEvalError("row_integrity")
    expected_digest = hashlib.sha256(slotted.expected_value.encode("ascii")).hexdigest()
    for outcome in row.outcomes:
        if (
            type(outcome.policy) is not str
            or type(outcome.response_sha256) is not str
            or _SHA256_PATTERN.fullmatch(outcome.response_sha256) is None
            or type(outcome.response_utf8_bytes) is not int
            or outcome.response_utf8_bytes < 0
            or outcome.response_utf8_bytes > 1_048_576
            or type(outcome.exact_match) is not bool
            or outcome.exact_match
            != (
                outcome.response_utf8_bytes == len(slotted.expected_value)
                and outcome.response_sha256 == expected_digest
            )
        ):
            raise LocalUpdateCausalEvalError("row_integrity")


def make_causal_eval_row_v1(
    case: FrozenCaseV1,
    seal: CaseDecisionSealV1,
    target_schedule: TargetCategoryScheduleV1,
    slotted: SlottedCaseV1,
    responses_by_policy: Mapping[str, str],
) -> CausalEvalRowV1:
    """Score one paired subject while retaining no raw answer in its row."""

    _validate_slotted_case(case, seal, target_schedule, slotted)
    if not isinstance(responses_by_policy, Mapping) or set(responses_by_policy) != set(
        POLICIES
    ):
        raise LocalUpdateCausalEvalError("responses")
    outcomes: list[PolicyOutcomeV1] = []
    for policy in policy_order_v1(case.case_index):
        response = responses_by_policy[policy]
        if type(response) is not str:
            raise LocalUpdateCausalEvalError("responses")
        try:
            response_bytes = response.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise LocalUpdateCausalEvalError("responses") from error
        if len(response_bytes) > 1_048_576:
            raise LocalUpdateCausalEvalError("responses")
        outcomes.append(
            PolicyOutcomeV1(
                policy=policy,
                response_sha256=hashlib.sha256(response_bytes).hexdigest(),
                response_utf8_bytes=len(response_bytes),
                exact_match=response == slotted.expected_value,
            )
        )
    row = CausalEvalRowV1(
        schema_version=SCHEMA_VERSION,
        case_index=case.case_index,
        case_id=case.case_id,
        frozen_case_sha256=frozen_case_sha256_v1(case),
        decision_seal_sha256=seal.seal_sha256,
        target_category_schedule_sha256=target_schedule.schedule_sha256,
        slotted_case_sha256=slotted.content_sha256,
        target_category=slotted.target_category,
        policy_schedule=policy_order_v1(case.case_index),
        outcomes=tuple(outcomes),
    )
    _validate_row(case, seal, target_schedule, slotted, row)
    return row


def validate_complete_causal_eval_v1(
    cases: tuple[FrozenCaseV1, ...],
    seals: tuple[CaseDecisionSealV1, ...],
    target_schedule: TargetCategoryScheduleV1,
    slotted_cases: tuple[SlottedCaseV1, ...],
    rows: tuple[CausalEvalRowV1, ...],
) -> None:
    """Require all and only the 64 registered paired rows, in fixed order."""

    if (
        type(cases) is not tuple
        or type(seals) is not tuple
        or type(slotted_cases) is not tuple
        or type(rows) is not tuple
        or not (
            len(cases) == len(seals) == len(slotted_cases) == len(rows) == CASE_COUNT
        )
    ):
        raise LocalUpdateCausalEvalError("run_completeness")
    _validate_frozen_cases_structure(cases)
    if cases != generate_frozen_cases_v1():
        raise LocalUpdateCausalEvalError("frozen_suite")
    _validate_target_category_schedule(cases, target_schedule)
    if (
        any(type(seal) is not CaseDecisionSealV1 for seal in seals)
        or any(type(item) is not SlottedCaseV1 for item in slotted_cases)
        or any(type(row) is not CausalEvalRowV1 for row in rows)
        or tuple(seal.case_index for seal in seals) != tuple(range(CASE_COUNT))
        or tuple(item.case_index for item in slotted_cases) != tuple(range(CASE_COUNT))
        or tuple(row.case_index for row in rows) != tuple(range(CASE_COUNT))
        or len({item.future_nonce for item in slotted_cases}) != CASE_COUNT
    ):
        raise LocalUpdateCausalEvalError("run_completeness")
    counts = {category: 0 for category in ENTRY_CATEGORIES}
    for case, seal, slotted, row in zip(
        cases,
        seals,
        slotted_cases,
        rows,
        strict=True,
    ):
        _validate_row(case, seal, target_schedule, slotted, row)
        counts[row.target_category] += 1
    if tuple(sorted(counts.items())) != tuple(sorted(CATEGORY_CASE_COUNTS)):
        raise LocalUpdateCausalEvalError("run_completeness")


def _validate_no_plaintext_row_structure(row: CausalEvalRowV1) -> None:
    if (
        type(row) is not CausalEvalRowV1
        or type(row.schema_version) is not int
        or row.schema_version != SCHEMA_VERSION
        or type(row.case_index) is not int
        or row.case_index not in range(CASE_COUNT)
        or type(row.case_id) is not str
        or _CASE_ID_PATTERN.fullmatch(row.case_id) is None
        or type(row.frozen_case_sha256) is not str
        or _SHA256_PATTERN.fullmatch(row.frozen_case_sha256) is None
        or row.case_id != f"lcase_{row.frozen_case_sha256[:24]}"
        or type(row.decision_seal_sha256) is not str
        or _SHA256_PATTERN.fullmatch(row.decision_seal_sha256) is None
        or type(row.target_category_schedule_sha256) is not str
        or _SHA256_PATTERN.fullmatch(row.target_category_schedule_sha256) is None
        or type(row.slotted_case_sha256) is not str
        or _SHA256_PATTERN.fullmatch(row.slotted_case_sha256) is None
        or type(row.target_category) is not str
        or row.target_category not in ENTRY_CATEGORIES
        or type(row.policy_schedule) is not tuple
        or row.policy_schedule != policy_order_v1(row.case_index)
        or type(row.outcomes) is not tuple
        or any(
            type(outcome) is not PolicyOutcomeV1
            or type(outcome.policy) is not str
            or type(outcome.response_sha256) is not str
            or _SHA256_PATTERN.fullmatch(outcome.response_sha256) is None
            or type(outcome.response_utf8_bytes) is not int
            or outcome.response_utf8_bytes < 0
            or outcome.response_utf8_bytes > 1_048_576
            or type(outcome.exact_match) is not bool
            for outcome in row.outcomes
        )
        or tuple(outcome.policy for outcome in row.outcomes) != row.policy_schedule
    ):
        raise LocalUpdateCausalEvalError("row_integrity")


def no_plaintext_run_wire_v1(rows: tuple[CausalEvalRowV1, ...]) -> bytes:
    """Canonical result projection containing no target or response strings."""

    if (
        type(rows) is not tuple
        or len(rows) != CASE_COUNT
        or any(type(row) is not CausalEvalRowV1 for row in rows)
        or tuple(row.case_index for row in rows) != tuple(range(CASE_COUNT))
    ):
        raise LocalUpdateCausalEvalError("run_completeness")
    for row in rows:
        _validate_no_plaintext_row_structure(row)
    counts = {
        category: sum(row.target_category == category for row in rows)
        for category in ENTRY_CATEGORIES
    }
    if (
        tuple(sorted(counts.items())) != tuple(sorted(CATEGORY_CASE_COUNTS))
        or len({row.case_id for row in rows}) != CASE_COUNT
        or len({row.frozen_case_sha256 for row in rows}) != CASE_COUNT
        or len({row.decision_seal_sha256 for row in rows}) != CASE_COUNT
        or len({row.target_category_schedule_sha256 for row in rows}) != 1
        or len({row.slotted_case_sha256 for row in rows}) != CASE_COUNT
    ):
        raise LocalUpdateCausalEvalError("run_completeness")
    value = {
        "case_count": CASE_COUNT,
        "case_seed_sha256": hashlib.sha256(CASE_SEED.encode("ascii")).hexdigest(),
        "policy_schedule_algorithm": POLICY_SCHEDULE_ALGORITHM,
        "rows": [_no_plaintext_row_value(row) for row in rows],
        "schema_version": SCHEMA_VERSION,
    }
    return _canonical_json_bytes(value)


def causal_eval_root_sha256_v1(
    cases: tuple[FrozenCaseV1, ...],
    seals: tuple[CaseDecisionSealV1, ...],
    target_schedule: TargetCategoryScheduleV1,
    slotted_cases: tuple[SlottedCaseV1, ...],
    rows: tuple[CausalEvalRowV1, ...],
) -> str:
    """Return a root only after private truth and all 64 bindings validate."""

    validate_complete_causal_eval_v1(
        cases,
        seals,
        target_schedule,
        slotted_cases,
        rows,
    )
    return hashlib.sha256(_RUN_ROOT_DOMAIN + no_plaintext_run_wire_v1(rows)).hexdigest()


def _exact_sign_test_p_value(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _category_control_role(category: str) -> str:
    return {
        "corrected": "exact_signal",
        "paraphrased_corrected": "parser_recall",
        "stable_placebo": "placebo",
        "corrupt_feedback": "adversarial_stress",
        "no_fact_feedback": "placebo",
    }[category]


def analyze_causal_eval_v1(
    cases: tuple[FrozenCaseV1, ...],
    seals: tuple[CaseDecisionSealV1, ...],
    target_schedule: TargetCategoryScheduleV1,
    slotted_cases: tuple[SlottedCaseV1, ...],
    rows: tuple[CausalEvalRowV1, ...],
) -> CausalEvalMetricsV1:
    """Validate the closed run, then compute paired and stratified metrics."""

    validate_complete_causal_eval_v1(
        cases,
        seals,
        target_schedule,
        slotted_cases,
        rows,
    )
    by_case = [
        {outcome.policy: outcome.exact_match for outcome in row.outcomes}
        for row in rows
    ]
    policy_accuracies = tuple(
        PolicyAccuracyV1(
            policy=policy,
            successes=sum(result[policy] for result in by_case),
            total=CASE_COUNT,
            accuracy=sum(result[policy] for result in by_case) / CASE_COUNT,
        )
        for policy in POLICIES
    )
    category_accuracies: list[CategoryAccuracyV1] = []
    for policy in POLICIES:
        for category in ENTRY_CATEGORIES:
            selected = [
                result[policy]
                for result, row in zip(by_case, rows, strict=True)
                if row.target_category == category
            ]
            category_accuracies.append(
                CategoryAccuracyV1(
                    policy=policy,
                    category=category,
                    successes=sum(selected),
                    total=len(selected),
                    accuracy=sum(selected) / len(selected),
                    control_role=_category_control_role(category),
                )
            )
    comparisons: list[PairedComparisonV1] = []
    for left, right in (
        ("feedback_latest", "noop"),
        ("feedback_latest", "latest_any"),
        ("noop", "latest_any"),
    ):
        differences = [int(result[left]) - int(result[right]) for result in by_case]
        wins = differences.count(1)
        losses = differences.count(-1)
        ties = differences.count(0)
        category_deltas = []
        for category in ENTRY_CATEGORIES:
            selected = [
                int(result[left]) - int(result[right])
                for result, row in zip(by_case, rows, strict=True)
                if row.target_category == category
            ]
            category_deltas.append(sum(selected) / len(selected))
        comparisons.append(
            PairedComparisonV1(
                left_policy=left,
                right_policy=right,
                wins=wins,
                losses=losses,
                ties=ties,
                mean_delta=sum(differences) / CASE_COUNT,
                macro_category_delta=(sum(category_deltas) / len(category_deltas)),
                discordant_count=wins + losses,
                exact_sign_test_p_value=_exact_sign_test_p_value(wins, losses),
            )
        )
    return CausalEvalMetricsV1(
        schema_version=SCHEMA_VERSION,
        case_count=CASE_COUNT,
        no_plaintext_root_sha256=causal_eval_root_sha256_v1(
            cases,
            seals,
            target_schedule,
            slotted_cases,
            rows,
        ),
        policy_accuracies=policy_accuracies,
        category_accuracies=tuple(category_accuracies),
        paired_comparisons=tuple(comparisons),
        aggregate_weighting="registered-target-prevalence-descriptive-v1",
        placebo_control_categories=PLACEBO_CONTROL_CATEGORIES,
        adversarial_stress_categories=ADVERSARIAL_STRESS_CATEGORIES,
    )
