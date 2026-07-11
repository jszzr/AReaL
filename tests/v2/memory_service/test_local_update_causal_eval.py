# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections import Counter

import pytest

from examples.memory_service import local_update_causal_eval as causal


def _decision_hash(case_index: int, policy: str) -> str:
    return hashlib.sha256(f"decision|{case_index}|{policy}".encode()).hexdigest()


def _future_nonce(case_index: int) -> str:
    return hashlib.sha256(f"future-after-seal|{case_index}".encode()).hexdigest()


def _category_nonce() -> str:
    return hashlib.sha256(b"category-after-all-updates-sealed").hexdigest()


@pytest.fixture(scope="module")
def reference_run():
    cases = causal.generate_frozen_cases_v1()
    target_schedule = causal.make_target_category_schedule_v1(
        cases,
        category_nonce=_category_nonce(),
    )
    seals = []
    slotted_cases = []
    rows = []
    for case in cases:
        seal = causal.make_case_decision_seal_v1(
            case,
            {
                policy: _decision_hash(case.case_index, policy)
                for policy in causal.POLICIES
            },
        )
        slotted = causal.slot_frozen_case_v1(
            case,
            seal,
            target_schedule,
            future_nonce=_future_nonce(case.case_index),
        )
        responses = {
            policy: causal.reference_policy_response_v1(
                case,
                seal,
                target_schedule,
                slotted,
                policy,
            )
            for policy in causal.POLICIES
        }
        rows.append(
            causal.make_causal_eval_row_v1(
                case,
                seal,
                target_schedule,
                slotted,
                responses,
            )
        )
        seals.append(seal)
        slotted_cases.append(slotted)
    return cases, tuple(seals), target_schedule, tuple(slotted_cases), tuple(rows)


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_mapping_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_mapping_keys(child)}
    return set()


def test_frozen_suite_has_exact_registered_shape_and_fingerprint() -> None:
    first = causal.generate_frozen_cases_v1()
    second = causal.generate_frozen_cases_v1()

    assert first == second
    assert len(first) == 64
    assert tuple(case.case_index for case in first) == tuple(range(64))
    assert all(not hasattr(case, "target_category") for case in first)
    assert (
        causal.frozen_suite_sha256_v1(first)
        == "4f096544014d7ef46006eef5359de38b6a4ab4a9a698971e1d02a8d092b4be1d"
    )
    for case in first:
        assert Counter(slot.category for slot in case.slots) == {
            "corrected": 3,
            "paraphrased_corrected": 1,
            "stable_placebo": 2,
            "corrupt_feedback": 1,
            "no_fact_feedback": 1,
        }
        assert tuple(slot.slot for slot in case.slots) == tuple(range(8))
        assert {
            event.sequence_no for slot in case.slots for event in slot.evidence
        } == set(range(16))


def test_generated_keys_and_unrelated_values_never_collide() -> None:
    cases = causal.generate_frozen_cases_v1()
    keys = [slot.key for case in cases for slot in case.slots]
    assert len(keys) == len(set(keys)) == 64 * 8

    # Equal tokens are intentional only inside one slot (for example, stable
    # truth equals its base and corrupt feedback repeats the same bad value).
    owners: dict[str, tuple[int, int]] = {}
    value_pattern = re.compile(r"[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5}")
    for case in cases:
        for slot in case.slots:
            values = {slot.base_value, slot.expected_value}
            values.update(
                token
                for event in slot.evidence
                for token in value_pattern.findall(event.payload)
            )
            for value in values:
                assert owners.setdefault(value, (case.case_index, slot.slot)) == (
                    case.case_index,
                    slot.slot,
                )


def test_category_semantics_expose_parser_recall_and_distinct_controls() -> None:
    case = causal.generate_frozen_cases_v1()[0]
    by_category = {slot.category: slot for slot in case.slots}

    corrected = by_category["corrected"]
    assert corrected.expected_value != corrected.base_value
    assert corrected.evidence[0].kind == "feedback"
    assert corrected.evidence[0].payload.endswith(corrected.expected_value)

    paraphrased = by_category["paraphrased_corrected"]
    assert paraphrased.expected_value != paraphrased.base_value
    assert " = " not in paraphrased.evidence[0].payload
    assert paraphrased.key in paraphrased.evidence[0].payload
    assert paraphrased.expected_value in paraphrased.evidence[0].payload

    stable = by_category["stable_placebo"]
    assert stable.expected_value == stable.base_value
    assert all(event.payload.endswith(stable.base_value) for event in stable.evidence)

    corrupt = by_category["corrupt_feedback"]
    assert corrupt.expected_value == corrupt.base_value
    assert not corrupt.evidence[0].payload.endswith(corrupt.expected_value)

    no_fact = by_category["no_fact_feedback"]
    assert no_fact.expected_value == no_fact.base_value
    assert no_fact.evidence[0].kind == "feedback"
    assert no_fact.key in no_fact.evidence[0].payload
    assert " = " not in no_fact.evidence[0].payload
    assert (
        re.findall(
            r"[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5}", no_fact.evidence[0].payload
        )
        == []
    )


def test_target_categories_are_assigned_only_by_post_update_nonce() -> None:
    cases = causal.generate_frozen_cases_v1()
    first = causal.make_target_category_schedule_v1(
        cases,
        category_nonce=_category_nonce(),
    )
    replay = causal.make_target_category_schedule_v1(
        cases,
        category_nonce=_category_nonce(),
    )
    changed = causal.make_target_category_schedule_v1(
        cases,
        category_nonce=hashlib.sha256(b"different-post-update-beacon").hexdigest(),
    )

    assert first == replay
    assert first != changed
    assert Counter(first.categories) == {
        "corrected": 24,
        "paraphrased_corrected": 8,
        "stable_placebo": 16,
        "corrupt_feedback": 8,
        "no_fact_feedback": 8,
    }
    assert first.schedule_sha256 != changed.schedule_sha256


def test_policy_schedule_is_frozen_three_by_two_williams_design() -> None:
    assert causal.POLICY_SCHEDULE_ALGORITHM == "logical-williams-3-policy-2-block-v1"
    first_block = tuple(causal.policy_order_v1(index) for index in range(6))
    assert first_block == (
        ("feedback_latest", "noop", "latest_any"),
        ("noop", "latest_any", "feedback_latest"),
        ("latest_any", "feedback_latest", "noop"),
        ("latest_any", "noop", "feedback_latest"),
        ("feedback_latest", "latest_any", "noop"),
        ("noop", "feedback_latest", "latest_any"),
    )
    assert all(set(row) == set(causal.POLICIES) for row in first_block)
    for position in range(3):
        assert Counter(row[position] for row in first_block) == {
            policy: 2 for policy in causal.POLICIES
        }
    assert causal.policy_order_v1(6) == first_block[0]


def test_future_nonce_slots_only_after_all_policy_decisions_are_sealed() -> None:
    cases = causal.generate_frozen_cases_v1()
    target_schedule = causal.make_target_category_schedule_v1(
        cases,
        category_nonce=_category_nonce(),
    )
    case = next(
        item
        for item, category in zip(cases, target_schedule.categories, strict=True)
        if category == "corrected"
    )
    decisions = {
        policy: _decision_hash(case.case_index, policy) for policy in causal.POLICIES
    }
    seal = causal.make_case_decision_seal_v1(case, decisions)
    # The integration runner derives this once from its post-global-seal
    # beacon.  This unit test supplies one fixed derived value; it deliberately
    # does not search the nonce space for a preferred slot.
    nonce = hashlib.sha256(b"post-global-seal-beacon|case-0").hexdigest()
    slotted = causal.slot_frozen_case_v1(
        case,
        seal,
        target_schedule,
        future_nonce=nonce,
    )
    replayed = causal.slot_frozen_case_v1(
        case,
        seal,
        target_schedule,
        future_nonce=nonce,
    )
    assert replayed == slotted
    assert slotted.target_category == "corrected"
    assert slotted.target_slot in {
        slot.slot for slot in case.slots if slot.category == "corrected"
    }
    assert slotted.query == (
        f"What is the current code for {slotted.target_key}? "
        "Reply with exactly the code or UNKNOWN."
    )

    changed_decisions = dict(decisions)
    changed_decisions["noop"] = hashlib.sha256(b"changed-decision").hexdigest()
    changed_seal = causal.make_case_decision_seal_v1(case, changed_decisions)
    assert changed_seal.seal_sha256 != seal.seal_sha256


def test_forged_or_incomplete_decision_seal_is_rejected() -> None:
    case = causal.generate_frozen_cases_v1()[0]
    decisions = {
        policy: _decision_hash(case.case_index, policy) for policy in causal.POLICIES
    }
    with pytest.raises(causal.LocalUpdateCausalEvalError) as missing:
        causal.make_case_decision_seal_v1(
            case,
            {key: value for key, value in decisions.items() if key != "noop"},
        )
    assert missing.value.reason == "decision_seal"

    seal = causal.make_case_decision_seal_v1(case, decisions)
    schedule = causal.make_target_category_schedule_v1(
        causal.generate_frozen_cases_v1(),
        category_nonce=_category_nonce(),
    )
    forged = dataclasses.replace(seal, seal_sha256="0" * 64)
    with pytest.raises(causal.LocalUpdateCausalEvalError) as invalid:
        causal.slot_frozen_case_v1(
            case,
            forged,
            schedule,
            future_nonce=hashlib.sha256(b"future").hexdigest(),
        )
    assert invalid.value.reason == "decision_seal"


def test_reference_policy_semantics_have_registered_category_truth_table(
    reference_run,
) -> None:
    cases, seals, target_schedule, slotted_cases, _rows = reference_run
    expected = {
        "corrected": {
            "feedback_latest": True,
            "noop": False,
            "latest_any": False,
        },
        "paraphrased_corrected": {
            "feedback_latest": False,
            "noop": False,
            "latest_any": False,
        },
        "stable_placebo": {
            "feedback_latest": True,
            "noop": True,
            "latest_any": True,
        },
        "corrupt_feedback": {
            "feedback_latest": False,
            "noop": True,
            "latest_any": False,
        },
        "no_fact_feedback": {
            "feedback_latest": True,
            "noop": True,
            "latest_any": False,
        },
    }
    for case, seal, slotted in zip(cases, seals, slotted_cases, strict=True):
        actual = {
            policy: causal.reference_policy_response_v1(
                case,
                seal,
                target_schedule,
                slotted,
                policy,
            )
            == slotted.expected_value
            for policy in causal.POLICIES
        }
        assert actual == expected[slotted.target_category]


def test_rows_and_run_root_contain_no_scorer_plaintext(reference_run) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    wire = causal.no_plaintext_run_wire_v1(rows)
    decoded = json.loads(wire)

    forbidden_keys = {
        "answer",
        "expected",
        "expected_value",
        "future_nonce",
        "query",
        "raw_response",
        "response",
        "target_key",
        "target_value",
        "truth",
    }
    assert _all_mapping_keys(decoded).isdisjoint(forbidden_keys)
    for case, slotted in zip(cases, slotted_cases, strict=True):
        assert slotted.target_key.encode() not in wire
        assert slotted.expected_value.encode() not in wire
        for slot in case.slots:
            assert slot.key.encode() not in wire
            assert slot.base_value.encode() not in wire
            assert slot.expected_value.encode() not in wire
    assert causal.causal_eval_root_sha256_v1(
        cases,
        seals,
        target_schedule,
        slotted_cases,
        rows,
    ) == ("ee19fb3327b34462964409717360d26007612e4442c46684ef4d6fc1e51e7da3")


def test_complete_reference_metrics_match_preregistered_effects(reference_run) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    result = causal.analyze_causal_eval_v1(
        cases,
        seals,
        target_schedule,
        slotted_cases,
        rows,
    )

    assert {
        item.policy: (item.successes, item.total, item.accuracy)
        for item in result.policy_accuracies
    } == {
        "feedback_latest": (48, 64, 0.75),
        "noop": (32, 64, 0.5),
        "latest_any": (16, 64, 0.25),
    }
    paired = {
        (item.left_policy, item.right_policy): item
        for item in result.paired_comparisons
    }
    feedback_noop = paired[("feedback_latest", "noop")]
    assert (feedback_noop.wins, feedback_noop.losses, feedback_noop.ties) == (
        24,
        8,
        32,
    )
    assert feedback_noop.mean_delta == 0.25
    assert feedback_noop.macro_category_delta == 0.0
    assert feedback_noop.discordant_count == 32
    assert feedback_noop.exact_sign_test_p_value == pytest.approx(
        2 * sum(math.comb(32, index) for index in range(9)) / 2**32,
        rel=0,
        abs=0,
    )
    assert paired[("feedback_latest", "latest_any")].mean_delta == 0.5
    assert paired[("noop", "latest_any")].mean_delta == 0.25


def test_control_roles_and_stratified_macro_tradeoff_are_preserved(
    reference_run,
) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    result = causal.analyze_causal_eval_v1(
        cases,
        seals,
        target_schedule,
        slotted_cases,
        rows,
    )

    assert result.aggregate_weighting == ("registered-target-prevalence-descriptive-v1")
    assert result.placebo_control_categories == (
        "stable_placebo",
        "no_fact_feedback",
    )
    assert result.adversarial_stress_categories == ("corrupt_feedback",)
    by_key = {(item.policy, item.category): item for item in result.category_accuracies}
    assert by_key[("feedback_latest", "corrupt_feedback")].accuracy == 0.0
    assert by_key[("feedback_latest", "no_fact_feedback")].accuracy == 1.0
    assert by_key[("feedback_latest", "paraphrased_corrected")].accuracy == 0.0
    assert by_key[("noop", "corrupt_feedback")].accuracy == 1.0
    assert by_key[("latest_any", "no_fact_feedback")].accuracy == 0.0
    for item in result.category_accuracies:
        if item.category in result.placebo_control_categories:
            assert item.control_role == "placebo"
        elif item.category in result.adversarial_stress_categories:
            assert item.control_role == "adversarial_stress"
    feedback_macro = sum(
        by_key[("feedback_latest", category)].accuracy
        for category in causal.ENTRY_CATEGORIES
    ) / len(causal.ENTRY_CATEGORIES)
    noop_macro = sum(
        by_key[("noop", category)].accuracy for category in causal.ENTRY_CATEGORIES
    ) / len(causal.ENTRY_CATEGORIES)
    assert feedback_macro == noop_macro == 0.6


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_exactly_64_rows_in_registered_order_are_required(
    reference_run, mutation
) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    if mutation == "missing":
        changed = rows[:-1]
    elif mutation == "duplicate":
        changed = rows[:-1] + (rows[-2],)
    else:
        changed = (rows[1], rows[0], *rows[2:])

    with pytest.raises(causal.LocalUpdateCausalEvalError) as error:
        causal.validate_complete_causal_eval_v1(
            cases,
            seals,
            target_schedule,
            slotted_cases,
            changed,
        )
    assert error.value.reason == "run_completeness"


def test_row_exact_match_cannot_be_flipped_after_scoring(reference_run) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    row = rows[0]
    outcome = row.outcomes[0]
    forged_outcome = dataclasses.replace(outcome, exact_match=not outcome.exact_match)
    forged_row = dataclasses.replace(
        row,
        outcomes=(forged_outcome, *row.outcomes[1:]),
    )
    forged_rows = (forged_row, *rows[1:])

    with pytest.raises(causal.LocalUpdateCausalEvalError) as error:
        causal.analyze_causal_eval_v1(
            cases,
            seals,
            target_schedule,
            slotted_cases,
            forged_rows,
        )
    assert error.value.reason == "row_integrity"


def test_row_cannot_be_rebound_to_another_category_or_slot(reference_run) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    forged_row = dataclasses.replace(rows[0], target_category="stable_placebo")

    with pytest.raises(causal.LocalUpdateCausalEvalError) as error:
        causal.validate_complete_causal_eval_v1(
            cases,
            seals,
            target_schedule,
            slotted_cases,
            (forged_row, *rows[1:]),
        )
    assert error.value.reason == "row_integrity"


def test_duplicate_future_nonce_is_rejected_for_complete_run(reference_run) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    # Re-slot case 1 with case 0's nonce, then build the internally consistent
    # row.  Per-case validation passes; run-level nonce uniqueness still fails.
    duplicate = causal.slot_frozen_case_v1(
        cases[1],
        seals[1],
        target_schedule,
        future_nonce=slotted_cases[0].future_nonce,
    )
    duplicate_row = causal.make_causal_eval_row_v1(
        cases[1],
        seals[1],
        target_schedule,
        duplicate,
        {
            policy: causal.reference_policy_response_v1(
                cases[1],
                seals[1],
                target_schedule,
                duplicate,
                policy,
            )
            for policy in causal.POLICIES
        },
    )
    changed_slotted = (slotted_cases[0], duplicate, *slotted_cases[2:])
    changed_rows = (rows[0], duplicate_row, *rows[2:])

    with pytest.raises(causal.LocalUpdateCausalEvalError) as error:
        causal.validate_complete_causal_eval_v1(
            cases,
            seals,
            target_schedule,
            changed_slotted,
            changed_rows,
        )
    assert error.value.reason == "run_completeness"


def test_no_plaintext_root_rejects_structurally_invalid_rows(reference_run) -> None:
    cases, seals, target_schedule, slotted_cases, rows = reference_run
    malformed = dataclasses.replace(
        rows[0],
        outcomes=(
            dataclasses.replace(rows[0].outcomes[0], response_utf8_bytes=True),
            *rows[0].outcomes[1:],
        ),
    )
    with pytest.raises(causal.LocalUpdateCausalEvalError) as error:
        causal.causal_eval_root_sha256_v1(
            cases,
            seals,
            target_schedule,
            slotted_cases,
            (malformed, *rows[1:]),
        )
    assert error.value.reason == "row_integrity"
