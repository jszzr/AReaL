# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import Counter

import pytest

from examples.memory_service import local_update_causal_eval as causal
from examples.memory_service import local_update_causal_runner as runner
from examples.memory_service import local_update_future_batch as future
from examples.memory_service import local_update_run_seal as run_seal


def test_one_real_case_binds_input_decisions_and_applied_releases(tmp_path) -> None:
    case = causal.generate_frozen_cases_v1()[0]
    captured = runner._capture_case_v1(case, tmp_path / "case")

    assert captured.case == case
    assert captured.policy_input.evidence_snapshot_id.startswith("esnap_")
    assert len(captured.policy_input.evidence_snapshot_members) == 24
    assert captured.input_sha256 == runner.policy.policy_input_sha256_v1(
        captured.policy_input
    )
    assert tuple(branch.name for branch in captured.branches) == causal.POLICIES
    assert {
        branch.name: (branch.applied.update_count, branch.applied.changed)
        for branch in captured.branches
    } == {
        "feedback_latest": (4, True),
        "noop": (0, False),
        "latest_any": (6, True),
    }
    assert captured.decision_seal == causal.make_case_decision_seal_v1(
        case,
        {
            branch.name: runner.policy.policy_decision_sha256_v1(branch.decision)
            for branch in captured.branches
        },
    )
    for branch in captured.branches:
        assert branch.applied.input_sha256 == captured.input_sha256
        assert branch.applied.base_release_id == captured.policy_input.base_release_id
        assert branch.applied.decision_sha256 == branch.decision_sha256
        assert branch.applied.evidence_root_sha256 == (
            runner.application.recompute_applied_policy_release_root_v1(branch.applied)
        )


@pytest.mark.parametrize("fail_at", [0, 63])
def test_any_capture_failure_consumes_no_beacon_or_future_execution(
    tmp_path,
    monkeypatch,
    fail_at: int,
) -> None:
    calls = {"capture": 0, "beacon": 0, "future": 0}

    def fail_capture(_case, _root):
        calls["capture"] += 1
        if calls["capture"] - 1 == fail_at:
            raise runner.LocalUpdateCausalRunError("injected_capture_failure")
        return object()

    def select_beacon(_external_beacon):
        calls["beacon"] += 1
        return b"\x01" * 32, "caller_supplied_external_unverified"

    def execute_future(*_args, **_kwargs):
        calls["future"] += 1
        raise AssertionError("future must not execute")

    monkeypatch.setattr(runner, "_capture_case_v1", fail_capture)
    monkeypatch.setattr(runner, "_select_beacon", select_beacon)
    monkeypatch.setattr(
        runner.future,
        "execute_verified_release_batch_v1",
        execute_future,
    )

    with pytest.raises(runner.LocalUpdateCausalRunError) as caught:
        runner.run_local_update_causal_experiment_v1(
            tmp_path / "failed-run",
            external_beacon=b"\x01" * 32,
        )
    assert caught.value.reason == "injected_capture_failure"
    assert calls == {"capture": fail_at + 1, "beacon": 0, "future": 0}


def test_durable_beacon_resumes_exactly_and_rejects_resampling(tmp_path) -> None:
    database = tmp_path / "seals.sqlite3"
    update = runner._seal_and_reopen(
        database,
        idempotency_key="complete-update-graph-v1",
        payload={"complete": True, "stage": "update"},
    )
    snapshot = runner._seal_and_reopen(
        database,
        idempotency_key="complete-database-snapshots-v1",
        payload={"complete": True, "stage": "snapshot"},
    )
    first = runner._resume_or_seal_beacon(
        database,
        snapshot_seal=snapshot,
        update_seal=update,
        external_beacon=b"\x11" * 32,
    )
    resumed = runner._resume_or_seal_beacon(
        database,
        snapshot_seal=snapshot,
        update_seal=update,
        external_beacon=b"\x11" * 32,
    )
    assert resumed == first

    with pytest.raises(runner.LocalUpdateCausalRunError) as caught:
        runner._resume_or_seal_beacon(
            database,
            snapshot_seal=snapshot,
            update_seal=update,
            external_beacon=b"\x22" * 32,
        )
    assert caught.value.reason == "beacon_conflict"

    malformed_database = tmp_path / "malformed-seals.sqlite3"
    malformed_update = runner._seal_and_reopen(
        malformed_database,
        idempotency_key="complete-update-graph-v1",
        payload={"complete": True, "stage": "update"},
    )
    malformed_snapshot = runner._seal_and_reopen(
        malformed_database,
        idempotency_key="complete-database-snapshots-v1",
        payload={"complete": True, "stage": "snapshot"},
    )
    runner._seal_and_reopen(
        malformed_database,
        idempotency_key="one-shot-future-beacon-v1",
        payload={"beacon_hex": (b"\x11" * 32).hex()},
    )
    with pytest.raises(runner.LocalUpdateCausalRunError) as malformed:
        runner._resume_or_seal_beacon(
            malformed_database,
            snapshot_seal=malformed_snapshot,
            update_seal=malformed_update,
            external_beacon=b"\x11" * 32,
        )
    assert malformed.value.reason == "durable_seal"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"external_beacon": b"short"},
        {"future_timeout_seconds": float("nan")},
        {"future_timeout_seconds": 0},
        {"future_timeout_seconds": 3601},
        {"future_timeout_seconds": 10**10000},
    ),
)
def test_invalid_run_inputs_fail_before_creating_any_state(tmp_path, kwargs) -> None:
    root = tmp_path / "must-not-exist"
    with pytest.raises(runner.LocalUpdateCausalRunError) as caught:
        runner.run_local_update_causal_experiment_v1(root, **kwargs)
    assert caught.value.reason == "closed_schema"
    assert not root.exists()


def test_existing_nonprivate_or_symlink_run_root_is_rejected(tmp_path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o700)
    loose.chmod(0o755)
    with pytest.raises(runner.LocalUpdateCausalRunError) as caught:
        runner.run_local_update_causal_experiment_v1(loose)
    assert caught.value.reason == "root_invalid"

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(runner.LocalUpdateCausalRunError) as caught:
        runner.run_local_update_causal_experiment_v1(link)
    assert caught.value.reason == "root_invalid"


def test_complete_64_case_real_path_seals_before_one_beacon_and_scores(
    tmp_path,
    monkeypatch,
) -> None:
    events: list[str] = []
    beacon_calls = 0
    original_seal = runner._seal_and_reopen
    original_future = future.execute_verified_release_batch_v1

    def seal(database_path, *, idempotency_key, payload):
        result = original_seal(
            database_path,
            idempotency_key=idempotency_key,
            payload=payload,
        )
        events.append(f"seal:{idempotency_key}")
        return result

    def select_beacon(_external_beacon):
        nonlocal beacon_calls
        beacon_calls += 1
        events.append("beacon")
        return b"\x42" * 32, "caller_supplied_external_unverified"

    def execute(items, *, timeout_seconds):
        events.append(f"future:{len(items)}")
        return original_future(items, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(runner, "_seal_and_reopen", seal)
    monkeypatch.setattr(runner, "_select_beacon", select_beacon)
    monkeypatch.setattr(
        runner.future,
        "execute_verified_release_batch_v1",
        execute,
    )
    monkeypatch.setattr(future, "_sample_execution_seed", lambda: b"\x24" * 32)

    root = tmp_path / "complete-run"
    result = runner.run_local_update_causal_experiment_v1(
        root,
        external_beacon=b"\x42" * 32,
        future_timeout_seconds=120,
    )

    assert beacon_calls == 1
    assert events == [
        "seal:complete-update-graph-v1",
        "seal:complete-database-snapshots-v1",
        "beacon",
        "seal:one-shot-future-beacon-v1",
        "seal:one-shot-future-schedule-v1",
        "future:192",
        "seal:complete-causal-result-v1",
    ]
    assert len(result.rows) == causal.CASE_COUNT
    assert {
        item.policy: (item.successes, item.total, item.accuracy)
        for item in result.metrics.policy_accuracies
    } == {
        "feedback_latest": (48, 64, 0.75),
        "noop": (32, 64, 0.5),
        "latest_any": (16, 64, 0.25),
    }
    primary = next(
        item
        for item in result.metrics.paired_comparisons
        if (item.left_policy, item.right_policy) == ("feedback_latest", "noop")
    )
    assert (primary.wins, primary.losses, primary.ties) == (24, 8, 32)
    assert primary.mean_delta == 0.25
    assert primary.macro_category_delta == 0.0
    assert primary.exact_sign_test_p_value == pytest.approx(0.0070003666914999485)

    seal_database = root / "causal-run-seals.sqlite3"
    assert result.update_seal == run_seal.get_local_update_run_seal(
        seal_database,
        scope="local-update-causal-v1",
        idempotency_key="complete-update-graph-v1",
    )
    assert result.schedule_seal == run_seal.get_local_update_run_seal(
        seal_database,
        scope="local-update-causal-v1",
        idempotency_key="one-shot-future-schedule-v1",
    )
    assert result.beacon_seal == run_seal.get_local_update_run_seal(
        seal_database,
        scope="local-update-causal-v1",
        idempotency_key="one-shot-future-beacon-v1",
    )
    assert result.snapshot_seal == run_seal.get_local_update_run_seal(
        seal_database,
        scope="local-update-causal-v1",
        idempotency_key="complete-database-snapshots-v1",
    )
    assert result.result_seal == run_seal.get_local_update_run_seal(
        seal_database,
        scope="local-update-causal-v1",
        idempotency_key="complete-causal-result-v1",
    )

    update_payload = json.loads(result.update_seal.canonical)["payload"]
    snapshot_payload = json.loads(result.snapshot_seal.canonical)["payload"]
    beacon_payload = json.loads(result.beacon_seal.canonical)["payload"]
    schedule_payload = json.loads(result.schedule_seal.canonical)["payload"]
    result_payload = json.loads(result.result_seal.canonical)["payload"]
    assert update_payload["complete"] is True
    assert len(update_payload["cases"]) == 64
    assert all(len(item["applications"]) == 3 for item in update_payload["cases"])
    diagnostics = {
        (item["policy"], item["category"]): item
        for item in update_payload["extraction_diagnostics"]
    }
    assert diagnostics[("feedback_latest", "corrected")]["correct_writes"] == 192
    assert (
        diagnostics[("feedback_latest", "paraphrased_corrected")][
            "missed_needed_updates"
        ]
        == 64
    )
    assert diagnostics[("feedback_latest", "corrupt_feedback")]["harmful_writes"] == 64
    assert diagnostics[("feedback_latest", "stable_placebo")]["safe_no_write"] == 128
    assert diagnostics[("feedback_latest", "no_fact_feedback")]["safe_no_write"] == 64
    assert snapshot_payload["complete"] is True
    assert len(snapshot_payload["database_snapshots"]) == 64
    assert beacon_payload["complete"] is True
    assert beacon_payload["beacon_hex"] == (b"\x42" * 32).hex()
    assert beacon_payload["beacon_source"] == ("caller_supplied_external_unverified")
    assert beacon_payload["snapshot_seal_id"] == result.snapshot_seal.seal_id
    assert schedule_payload["complete"] is True
    assert schedule_payload["query_count"] == 192
    assert len(schedule_payload["cases"]) == 64
    assert schedule_payload["target_category_schedule_sha256"] == (
        result.target_category_schedule.schedule_sha256
    )
    assert Counter(result.target_category_schedule.categories) == {
        "corrected": 24,
        "paraphrased_corrected": 8,
        "stable_placebo": 16,
        "corrupt_feedback": 8,
        "no_fact_feedback": 8,
    }
    assert result_payload["complete"] is True
    assert len(result_payload["execution_receipts"]) == 192
    assert result_payload["result_root_sha256"] == (
        result.metrics.no_plaintext_root_sha256
    )

    # Durable receipts omit plaintext keys, values, queries, and responses.
    # The public randomness beacon is intentionally retained so target
    # derivation can be reproduced.  Hashes over the small deterministic
    # corpus are an integrity mechanism, not a secrecy claim.
    canonical = b"".join(
        (
            result.update_seal.canonical,
            result.snapshot_seal.canonical,
            result.beacon_seal.canonical,
            result.schedule_seal.canonical,
            result.result_seal.canonical,
        )
    )
    for case in causal.generate_frozen_cases_v1():
        for slot in case.slots:
            assert slot.key.encode() not in canonical
            assert slot.base_value.encode() not in canonical
            assert slot.expected_value.encode() not in canonical
            assert (
                f"What is the current code for {slot.key}? ".encode() not in canonical
            )
            for evidence in slot.evidence:
                assert evidence.payload.encode() not in canonical

    # The private SQLite working data deliberately contains plaintext.  The
    # no-plaintext claim applies only to durable seal projections, not local
    # files protected by the enforced owner-only directories.
    first_snapshot = root / "future-database-snapshots" / "case-00.sqlite3"
    assert b"project-" in first_snapshot.read_bytes()

    events.clear()
    resumed = runner.run_local_update_causal_experiment_v1(
        root,
        external_beacon=b"\x42" * 32,
        future_timeout_seconds=120,
    )
    assert resumed == result
    assert events == [
        "seal:complete-update-graph-v1",
        "seal:complete-database-snapshots-v1",
        "seal:one-shot-future-schedule-v1",
        "future:192",
        "seal:complete-causal-result-v1",
    ]
