# SPDX-License-Identifier: Apache-2.0

"""Run the complete local Memory update-policy causal experiment.

The runner closes the boundaries intentionally left generic by the pure
evaluator:

* every frozen case is materialized through the real SQLite Memory store;
* one exact PolicyInput is shared by all three policies;
* each decision is recomputed, applied, and bound to its explicit release;
* all 64 x 3 application receipts are durably sealed before target selection;
* 64 consistent read-only database snapshots are sealed before any beacon;
* one run-level beacon is sampled once and deterministically derives all 64
  target categories and future slots, preventing per-case grinding by the
  honest runner;
* that schedule is durably sealed before an isolated child sees any query;
* verified child receipts are joined to scorer truth only after the entire
  batch passes provenance, rendering, state-isolation, and database checks.

This is a reproducible causal test of three fixed local policies, not model
training and not a malicious-code sandbox.  The scripted consumer lets the
experiment isolate Memory extraction/update quality from language-model noise.
Spark or a real model can later replace only the final consumer stage while
preserving the sealed update and scheduling protocol.

The built-in local CSPRNG mode is a smoke-test boundary.  A caller-supplied
beacon is recorded but not independently authenticated.  Publishable research
must anchor the update root before obtaining an externally verifiable future
beacon (or commit-reveal), and arbitrary new policies require a private
post-commit manifest because this deterministic suite is public.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import stat
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from examples.memory_service import local_update_causal_eval as causal
from examples.memory_service import local_update_future_batch as future
from examples.memory_service import local_update_policy_apply as application
from examples.memory_service import local_update_policy_eval as policy
from examples.memory_service import local_update_run_seal as run_seal
from examples.memory_service import scoped_codebook_eval as harness

from areal.v2.memory_service import (
    CandidateProposal,
    EvidenceEvent,
    EvidenceKind,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

__all__ = [
    "LocalUpdateCausalRunError",
    "LocalUpdateCausalRunReceiptV1",
    "run_local_update_causal_experiment_v1",
]


_SCHEMA_VERSION = 1
_EXPERIMENT = "areal-memory-local-update-64-case-v1"
_SEAL_SCOPE = "local-update-causal-v1"
_UPDATE_SEAL_KEY = "complete-update-graph-v1"
_SNAPSHOT_SEAL_KEY = "complete-database-snapshots-v1"
_BEACON_SEAL_KEY = "one-shot-future-beacon-v1"
_SCHEDULE_SEAL_KEY = "one-shot-future-schedule-v1"
_RESULT_SEAL_KEY = "complete-causal-result-v1"
_CASE_NAMESPACE = "local-update-causal-v1"
_OBSERVED_BASE = datetime(2026, 7, 12, tzinfo=UTC)
_CASE_NONCE_DOMAIN = b"areal-memory-local-update-case-nonce-v1\0"
_CATEGORY_NONCE_DOMAIN = b"areal-memory-local-update-category-nonce-v1\0"
_EXPECTED_UPDATE_COUNTS = {
    "feedback_latest": 4,
    "noop": 0,
    "latest_any": 6,
}


class LocalUpdateCausalRunError(RuntimeError):
    """Stable reason for failing the all-or-nothing causal run."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("causal-run reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class LocalUpdateCausalRunReceiptV1:
    """No-plaintext durable roots and validated metrics for one complete run."""

    schema_version: int
    experiment: str
    update_seal: run_seal.LocalUpdateRunSeal
    snapshot_seal: run_seal.LocalUpdateRunSeal
    beacon_seal: run_seal.LocalUpdateRunSeal
    schedule_seal: run_seal.LocalUpdateRunSeal
    result_seal: run_seal.LocalUpdateRunSeal
    target_category_schedule: causal.TargetCategoryScheduleV1
    metrics: causal.CausalEvalMetricsV1
    rows: tuple[causal.CausalEvalRowV1, ...]


@dataclass(frozen=True, slots=True)
class _PolicyBranch:
    name: str
    decision: policy.PolicyDecisionV1
    decision_sha256: str
    applied: application.AppliedPolicyReleaseV1


@dataclass(frozen=True, slots=True)
class _CapturedCase:
    case: causal.FrozenCaseV1
    database_path: str
    scope: MemoryScope
    policy_input: policy.PolicyInputV1
    input_sha256: str
    decision_seal: causal.CaseDecisionSealV1
    branches: tuple[_PolicyBranch, ...]
    database_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _FutureBinding:
    logical_index: int
    case_index: int
    policy: str
    application_root_sha256: str
    database_snapshot_sha256: str
    scope_sha256: str
    release_id: str
    release_content_sha256: str
    query_sha256: str


def _fail(reason: str, error: BaseException | None = None) -> None:
    failure = LocalUpdateCausalRunError(reason)
    if error is None:
        raise failure
    raise failure from error


def _is_sha256_text(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_private_directory(path: Path, *, reason: str) -> Path:
    try:
        state = path.lstat()
        resolved = path.resolve(strict=True)
        resolved_state = resolved.stat()
    except (OSError, RuntimeError, ValueError) as error:
        _fail(reason, error)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or state.st_uid != os.geteuid()
        or state.st_mode & 0o077
        or (state.st_dev, state.st_ino)
        != (resolved_state.st_dev, resolved_state.st_ino)
    ):
        _fail(reason)
    return resolved


def _branch_by_name(captured: _CapturedCase) -> dict[str, _PolicyBranch]:
    result = {branch.name: branch for branch in captured.branches}
    if set(result) != set(causal.POLICIES) or len(result) != len(causal.POLICIES):
        _fail("branch_integrity")
    return result


def _append_base_graph(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    case: causal.FrozenCaseV1,
) -> tuple[str, ...]:
    revision_ids: list[str] = []
    for slot in case.slots:
        payload = f"{slot.key} = {slot.base_value}"
        evidence = store.append(
            EvidenceEvent(
                scope=scope,
                session_id=f"{case.case_id}-base-session",
                run_id=f"{case.case_id}-base-run",
                sequence_no=slot.slot,
                kind=EvidenceKind.USER_MESSAGE,
                payload=payload,
                observed_at=_OBSERVED_BASE + timedelta(seconds=-16 + slot.slot),
                idempotency_key=f"{case.case_id}-base-evidence-{slot.slot:02d}",
            )
        )
        candidate = store.append_candidate(
            CandidateProposal(
                scope=scope,
                content=payload,
                evidence_ids=(evidence.evidence_id,),
                idempotency_key=f"{case.case_id}-base-candidate-{slot.slot:02d}",
            )
        )
        revision = store.append_revision(
            RevisionProposal(
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=RevisionOperation.ADD,
                parent_revision_id=None,
                idempotency_key=f"{case.case_id}-base-revision-{slot.slot:02d}",
            )
        )
        revision_ids.append(revision.revision_id)
    return tuple(revision_ids)


def _append_update_evidence(
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    case: causal.FrozenCaseV1,
) -> None:
    evidence = sorted(
        (item for slot in case.slots for item in slot.evidence),
        key=lambda item: (item.observed_offset_seconds, item.sequence_no),
    )
    if tuple(item.sequence_no for item in evidence) != tuple(range(16)):
        _fail("frozen_case")
    for item in evidence:
        try:
            kind = EvidenceKind(item.kind)
        except ValueError as error:
            _fail("frozen_case", error)
        store.append(
            EvidenceEvent(
                scope=scope,
                session_id=f"{case.case_id}-update-session",
                run_id=f"{case.case_id}-update-run",
                sequence_no=item.sequence_no,
                kind=kind,
                payload=item.payload,
                observed_at=_OBSERVED_BASE
                + timedelta(seconds=item.observed_offset_seconds),
                idempotency_key=(
                    f"{case.case_id}-update-evidence-{item.sequence_no:02d}"
                ),
            )
        )


def _validate_applied_branch(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    policy_input: policy.PolicyInputV1,
    name: str,
    decision: policy.PolicyDecisionV1,
    applied: application.AppliedPolicyReleaseV1,
) -> None:
    input_sha256 = policy.policy_input_sha256_v1(policy_input)
    decision_sha256 = policy.policy_decision_sha256_v1(decision)
    try:
        expected_root = application.recompute_applied_policy_release_root_v1(applied)
        release = store.get_release(scope, applied.release_id)
    except Exception as error:
        _fail("application_integrity", error)
    if (
        decision.policy != name
        or decision.input_sha256 != input_sha256
        or applied.policy != name
        or applied.input_sha256 != input_sha256
        or applied.decision_sha256 != decision_sha256
        or applied.base_release_id != policy_input.base_release_id
        or applied.base_release_content_sha256
        != policy_input.base_release_content_sha256
        or applied.release_id != release.release_id
        or applied.release_content_sha256 != release.content_hash
        or applied.revision_ids != release.manifest.revision_ids
        or applied.evidence_root_sha256 != expected_root
        or applied.update_count != _EXPECTED_UPDATE_COUNTS[name]
        or applied.changed is (name == "noop")
    ):
        _fail("application_integrity")


def _capture_case_v1(case: causal.FrozenCaseV1, case_root: Path) -> _CapturedCase:
    try:
        case_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        case_root = _require_private_directory(case_root, reason="case_capture")
        database_path = case_root / "memory.sqlite3"
        store = SQLiteMemoryStore(database_path)
        scope = MemoryScope(
            tenant_id="memory-eval",
            namespace=_CASE_NAMESPACE,
            subject_id=case.case_id,
        )
        base_revision_ids = _append_base_graph(store, scope, case)
        base_release = store.append_release(
            ReleaseManifest(scope=scope, revision_ids=base_revision_ids),
            idempotency_key=f"{case.case_id}-base-release",
        )
        _append_update_evidence(store, scope, case)
        policy_input = policy.make_policy_input_v1(
            store=store,
            scope=scope,
            base_release_id=base_release.release_id,
            cutoff=_OBSERVED_BASE + timedelta(seconds=16),
            evidence_snapshot_idempotency_key=f"{case.case_id}-policy-snapshot",
        )
        input_sha256 = policy.policy_input_sha256_v1(policy_input)
        branches: list[_PolicyBranch] = []
        decisions: dict[str, str] = {}
        for name in causal.POLICIES:
            decision = policy.run_local_update_policy_v1(name, policy_input)
            decision_sha256 = policy.policy_decision_sha256_v1(decision)
            applied = application.apply_local_update_decision_v1(
                store=store,
                scope=scope,
                policy_input=policy_input,
                decision=decision,
                expected_policy=name,
            )
            _validate_applied_branch(
                store=store,
                scope=scope,
                policy_input=policy_input,
                name=name,
                decision=decision,
                applied=applied,
            )
            branches.append(
                _PolicyBranch(
                    name=name,
                    decision=decision,
                    decision_sha256=decision_sha256,
                    applied=applied,
                )
            )
            decisions[name] = decision_sha256
        decision_seal = causal.make_case_decision_seal_v1(case, decisions)
    except LocalUpdateCausalRunError:
        raise
    except Exception as error:
        _fail("case_capture", error)
    captured = _CapturedCase(
        case=case,
        database_path=str(database_path.resolve(strict=True)),
        scope=scope,
        policy_input=policy_input,
        input_sha256=input_sha256,
        decision_seal=decision_seal,
        branches=tuple(branches),
    )
    if tuple(branch.name for branch in captured.branches) != causal.POLICIES:
        _fail("branch_integrity")
    return captured


def _application_payload(branch: _PolicyBranch) -> dict[str, object]:
    applied = branch.applied
    return {
        "application_id": applied.application_id,
        "application_root_sha256": applied.evidence_root_sha256,
        "changed": applied.changed,
        "decision_sha256": branch.decision_sha256,
        "policy": branch.name,
        "release_content_sha256": applied.release_content_sha256,
        "release_id": applied.release_id,
        "revision_count": applied.result_revision_count,
        "update_count": applied.update_count,
    }


def _extraction_diagnostics(
    captured: tuple[_CapturedCase, ...],
) -> list[dict[str, object]]:
    counters = {
        (name, category): {
            "correct_writes": 0,
            "harmful_writes": 0,
            "missed_needed_updates": 0,
            "needed_updates": 0,
            "opportunities": 0,
            "safe_no_write": 0,
            "writes": 0,
        }
        for name in causal.POLICIES
        for category in causal.ENTRY_CATEGORIES
    }
    for item in captured:
        slots_by_key = {slot.key: slot for slot in item.case.slots}
        if len(slots_by_key) != len(item.case.slots):
            _fail("frozen_case")
        for branch in item.branches:
            updates = {update.key: update for update in branch.decision.updates}
            if len(updates) != len(branch.decision.updates) or not set(updates) <= set(
                slots_by_key
            ):
                _fail("decision_integrity")
            for slot in item.case.slots:
                counter = counters[(branch.name, slot.category)]
                counter["opportunities"] += 1
                needed = slot.expected_value != slot.base_value
                if needed:
                    counter["needed_updates"] += 1
                update = updates.get(slot.key)
                if update is None:
                    if needed:
                        counter["missed_needed_updates"] += 1
                    else:
                        counter["safe_no_write"] += 1
                    continue
                counter["writes"] += 1
                if update.value == slot.expected_value:
                    counter["correct_writes"] += 1
                else:
                    counter["harmful_writes"] += 1
                    if needed:
                        counter["missed_needed_updates"] += 1
    return [
        {"category": category, "policy": name, **counters[(name, category)]}
        for name in causal.POLICIES
        for category in causal.ENTRY_CATEGORIES
    ]


def _update_seal_payload(
    cases: tuple[causal.FrozenCaseV1, ...],
    captured: tuple[_CapturedCase, ...],
) -> dict[str, object]:
    if (
        len(cases) != causal.CASE_COUNT
        or len(captured) != causal.CASE_COUNT
        or tuple(item.case.case_index for item in captured)
        != tuple(range(causal.CASE_COUNT))
    ):
        _fail("run_completeness")
    rows: list[dict[str, object]] = []
    for case, item in zip(cases, captured, strict=True):
        value = item.policy_input
        if case != item.case or item.input_sha256 != policy.policy_input_sha256_v1(
            value
        ):
            _fail("input_integrity")
        branch_by_name = _branch_by_name(item)
        expected_decisions = {
            name: policy.policy_decision_sha256_v1(branch_by_name[name].decision)
            for name in causal.POLICIES
        }
        expected_seal = causal.make_case_decision_seal_v1(
            case,
            expected_decisions,
        )
        if item.decision_seal != expected_seal:
            _fail("decision_integrity")
        rows.append(
            {
                "applications": [
                    _application_payload(branch_by_name[name])
                    for name in causal.POLICIES
                ],
                "base_release_content_sha256": value.base_release_content_sha256,
                "base_release_id": value.base_release_id,
                "case_id": case.case_id,
                "case_index": case.case_index,
                "decision_seal_sha256": item.decision_seal.seal_sha256,
                "evidence_snapshot_content_hash": (
                    value.evidence_snapshot_content_hash
                ),
                "evidence_snapshot_id": value.evidence_snapshot_id,
                "frozen_case_sha256": causal.frozen_case_sha256_v1(case),
                "input_sha256": item.input_sha256,
                "policy_scope_token": value.policy_scope_token,
            }
        )
    return {
        "case_count": causal.CASE_COUNT,
        "cases": rows,
        "complete": True,
        "experiment": _EXPERIMENT,
        "extraction_diagnostics": _extraction_diagnostics(captured),
        "frozen_suite_sha256": causal.frozen_suite_sha256_v1(cases),
        "policies": list(causal.POLICIES),
        "analysis_policy": {
            "aggregate_interpretation": (
                "registered_prevalence_descriptive_not_population_inference"
            ),
            "left_policy": "feedback_latest",
            "primary_metrics": [
                "category_stratified_accuracy",
                "macro_category_delta",
                "corrected_benefit",
                "corrupt_feedback_harm",
                "paraphrased_corrected_recall",
            ],
            "right_policy": "noop",
            "sign_test_role": "fixed_corpus_diagnostic_only",
            "weighted_delta_identity": (
                "p(corrected)-p(corrupt_feedback); other strata tie"
            ),
        },
        "schema_version": _SCHEMA_VERSION,
        "target_category_assignment_algorithm": (
            "beacon-hash-rank-cases-zip-preregistered-multiset-v1"
        ),
        "target_category_counts": [
            {"category": category, "count": count}
            for category, count in causal.CATEGORY_CASE_COUNTS
        ],
        "target_slot_assignment_algorithm": (
            "domain-separated-hash-rank-eligible-slots-v1"
        ),
    }


def _seal_and_reopen(
    database_path: Path,
    *,
    idempotency_key: str,
    payload: dict[str, object],
) -> run_seal.LocalUpdateRunSeal:
    try:
        sealed = run_seal.seal_local_update_run(
            database_path,
            scope=_SEAL_SCOPE,
            idempotency_key=idempotency_key,
            payload=payload,
        )
        reopened = run_seal.get_local_update_run_seal(
            database_path,
            scope=_SEAL_SCOPE,
            idempotency_key=idempotency_key,
        )
    except Exception as error:
        _fail("durable_seal", error)
    if sealed != reopened:
        _fail("durable_seal")
    return reopened


def _get_optional_seal(
    database_path: Path,
    *,
    idempotency_key: str,
) -> run_seal.LocalUpdateRunSeal | None:
    try:
        return run_seal.get_local_update_run_seal(
            database_path,
            scope=_SEAL_SCOPE,
            idempotency_key=idempotency_key,
        )
    except run_seal.LocalUpdateRunSealNotFoundError:
        return None
    except Exception as error:
        _fail("durable_seal", error)


def _sealed_payload(value: run_seal.LocalUpdateRunSeal) -> dict[str, object]:
    try:
        envelope = json.loads(value.canonical)
        payload = envelope["payload"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        _fail("durable_seal", error)
    if type(payload) is not dict or any(type(key) is not str for key in payload):
        _fail("durable_seal")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file_sha256(path: Path) -> str:
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o077
        ):
            _fail("database_snapshot")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = path.lstat()
    except LocalUpdateCausalRunError:
        raise
    except OSError as error:
        _fail("database_snapshot", error)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or (after.st_uid != os.geteuid() or after.st_nlink != 1 or after.st_mode & 0o077):
        _fail("database_snapshot")
    return digest.hexdigest()


def _sqlite_backup(source_path: Path, destination_path: Path) -> None:
    """Create one consistent private SQLite snapshot before future entropy."""

    staging = destination_path.with_name(
        f".{destination_path.name}.{secrets.token_hex(16)}.tmp"
    )
    descriptor: int | None = None
    source: sqlite3.Connection | None = None
    destination: sqlite3.Connection | None = None
    primary_error: BaseException | None = None
    try:
        descriptor = os.open(
            staging,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(descriptor)
        descriptor = None
        source = sqlite3.connect(
            source_path.resolve(strict=True).as_uri() + "?mode=ro",
            uri=True,
        )
        source.execute("PRAGMA query_only = ON")
        source.execute("BEGIN")
        destination = sqlite3.connect(staging)
        source.backup(destination)
        if destination.execute("PRAGMA quick_check").fetchone() != ("ok",):
            _fail("database_snapshot")
        destination.execute("PRAGMA journal_mode = DELETE")
        destination.commit()
        destination.close()
        destination = None
        source.rollback()
        source.close()
        source = None
        descriptor = os.open(staging, os.O_RDONLY)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.chmod(staging, 0o400)
        os.replace(staging, destination_path)
        _fsync_directory(destination_path.parent)
    except BaseException as error:
        if isinstance(error, LocalUpdateCausalRunError):
            primary_error = error
            raise
        if isinstance(error, (OSError, RuntimeError, sqlite3.Error, ValueError)):
            primary_error = LocalUpdateCausalRunError("database_snapshot")
            raise primary_error from error
        primary_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
        if destination is not None:
            try:
                destination.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if source is not None:
            try:
                source.close()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            staging.unlink(missing_ok=True)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note(
                        "snapshot cleanup failed after primary error: "
                        f"{type(error).__name__}: {error}"
                    )
            else:
                failure = LocalUpdateCausalRunError("database_snapshot")
                for error in cleanup_errors[1:]:
                    failure.add_note(
                        "additional snapshot cleanup failure: "
                        f"{type(error).__name__}: {error}"
                    )
                raise failure from cleanup_errors[0]


def _freeze_case_databases_v1(
    run_root: Path,
    captured: tuple[_CapturedCase, ...],
) -> tuple[_CapturedCase, ...]:
    snapshot_root = run_root / "future-database-snapshots"
    try:
        snapshot_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        _fail("database_snapshot", error)
    snapshot_root = _require_private_directory(
        snapshot_root,
        reason="database_snapshot",
    )

    frozen: list[_CapturedCase] = []
    for item in captured:
        destination = snapshot_root / f"case-{item.case.case_index:02d}.sqlite3"
        _sqlite_backup(Path(item.database_path), destination)
        content_sha256 = _regular_file_sha256(destination)
        try:
            snapshot_store = SQLiteMemoryStore(destination)
            for branch in item.branches:
                _validate_applied_branch(
                    store=snapshot_store,
                    scope=item.scope,
                    policy_input=item.policy_input,
                    name=branch.name,
                    decision=branch.decision,
                    applied=branch.applied,
                )
        except LocalUpdateCausalRunError:
            raise
        except Exception as error:
            _fail("database_snapshot", error)
        frozen.append(
            replace(
                item,
                database_path=str(destination.resolve(strict=True)),
                database_sha256=content_sha256,
            )
        )
    result = tuple(frozen)
    if (
        len(result) != causal.CASE_COUNT
        or any(item.database_sha256 is None for item in result)
        or len({item.database_sha256 for item in result}) != causal.CASE_COUNT
    ):
        _fail("database_snapshot")
    return result


def _snapshot_seal_payload(
    *,
    captured: tuple[_CapturedCase, ...],
    update_seal: run_seal.LocalUpdateRunSeal,
) -> dict[str, object]:
    if len(captured) != causal.CASE_COUNT or any(
        item.database_sha256 is None for item in captured
    ):
        _fail("database_snapshot")
    return {
        "case_count": causal.CASE_COUNT,
        "complete": True,
        "database_snapshots": [
            {
                "case_index": item.case.case_index,
                "database_snapshot_sha256": item.database_sha256,
            }
            for item in captured
        ],
        "experiment": _EXPERIMENT,
        "schema_version": _SCHEMA_VERSION,
        "update_seal_content_hash": update_seal.content_hash,
        "update_seal_id": update_seal.seal_id,
    }


def _beacon_seal_payload(
    *,
    snapshot_seal: run_seal.LocalUpdateRunSeal,
    update_seal: run_seal.LocalUpdateRunSeal,
    beacon: bytes,
    beacon_source: str,
) -> dict[str, object]:
    if (
        type(beacon) is not bytes
        or len(beacon) != 32
        or beacon_source
        not in {"local_csprng_smoke_test", "caller_supplied_external_unverified"}
    ):
        _fail("beacon_invalid")
    return {
        # The beacon is public randomness, not scorer truth.  Persisting it
        # makes target derivation independently reproducible and prevents a
        # hash-only receipt from becoming unverifiable after process exit.
        "beacon_hex": beacon.hex(),
        "beacon_sha256": hashlib.sha256(beacon).hexdigest(),
        "beacon_source": beacon_source,
        "complete": True,
        "experiment": _EXPERIMENT,
        "schema_version": _SCHEMA_VERSION,
        "snapshot_seal_content_hash": snapshot_seal.content_hash,
        "snapshot_seal_id": snapshot_seal.seal_id,
        "update_seal_content_hash": update_seal.content_hash,
        "update_seal_id": update_seal.seal_id,
    }


def _select_beacon(external_beacon: bytes | None) -> tuple[bytes, str]:
    if external_beacon is None:
        return secrets.token_bytes(32), "local_csprng_smoke_test"
    if type(external_beacon) is not bytes or len(external_beacon) != 32:
        _fail("beacon_invalid")
    return external_beacon, "caller_supplied_external_unverified"


def _resume_or_seal_beacon(
    seal_database: Path,
    *,
    snapshot_seal: run_seal.LocalUpdateRunSeal,
    update_seal: run_seal.LocalUpdateRunSeal,
    external_beacon: bytes | None,
) -> tuple[bytes, str, run_seal.LocalUpdateRunSeal]:
    existing = _get_optional_seal(
        seal_database,
        idempotency_key=_BEACON_SEAL_KEY,
    )
    if existing is not None:
        payload = _sealed_payload(existing)
        try:
            beacon = bytes.fromhex(payload["beacon_hex"])
            source = payload["beacon_source"]
        except (KeyError, TypeError, ValueError) as error:
            _fail("durable_seal", error)
        if type(source) is not str or len(beacon) != 32:
            _fail("beacon_conflict")
        expected_payload = _beacon_seal_payload(
            snapshot_seal=snapshot_seal,
            update_seal=update_seal,
            beacon=beacon,
            beacon_source=source,
        )
        if payload != expected_payload or (
            external_beacon is not None and external_beacon != beacon
        ):
            _fail("beacon_conflict")
        return beacon, source, existing

    beacon, source = _select_beacon(external_beacon)
    sealed = _seal_and_reopen(
        seal_database,
        idempotency_key=_BEACON_SEAL_KEY,
        payload=_beacon_seal_payload(
            snapshot_seal=snapshot_seal,
            update_seal=update_seal,
            beacon=beacon,
            beacon_source=source,
        ),
    )
    return beacon, source, sealed


def _case_nonce(
    *,
    beacon: bytes,
    update_seal: run_seal.LocalUpdateRunSeal,
    frozen_suite_sha256: str,
    case_index: int,
) -> str:
    return hashlib.sha256(
        _CASE_NONCE_DOMAIN
        + beacon
        + bytes.fromhex(update_seal.content_hash)
        + bytes.fromhex(frozen_suite_sha256)
        + case_index.to_bytes(8, "big")
    ).hexdigest()


def _make_future_schedule(
    *,
    cases: tuple[causal.FrozenCaseV1, ...],
    captured: tuple[_CapturedCase, ...],
    update_seal: run_seal.LocalUpdateRunSeal,
    snapshot_seal: run_seal.LocalUpdateRunSeal,
    beacon_seal: run_seal.LocalUpdateRunSeal,
    beacon: bytes,
) -> tuple[
    causal.TargetCategoryScheduleV1,
    tuple[causal.SlottedCaseV1, ...],
    tuple[future.FutureReleaseQueryV1, ...],
    tuple[_FutureBinding, ...],
    dict[str, object],
]:
    suite_sha256 = causal.frozen_suite_sha256_v1(cases)
    category_nonce = hashlib.sha256(
        _CATEGORY_NONCE_DOMAIN
        + beacon
        + bytes.fromhex(update_seal.content_hash)
        + bytes.fromhex(suite_sha256)
    ).hexdigest()
    target_schedule = causal.make_target_category_schedule_v1(
        cases,
        category_nonce=category_nonce,
    )
    slotted_cases: list[causal.SlottedCaseV1] = []
    queries: list[future.FutureReleaseQueryV1] = []
    bindings: list[_FutureBinding] = []
    schedule_rows: list[dict[str, object]] = []
    seen_nonces: set[str] = set()
    for case, item in zip(cases, captured, strict=True):
        if item.database_sha256 is None:
            _fail("database_snapshot")
        nonce = _case_nonce(
            beacon=beacon,
            update_seal=update_seal,
            frozen_suite_sha256=suite_sha256,
            case_index=case.case_index,
        )
        if nonce in seen_nonces:
            _fail("nonce_collision")
        seen_nonces.add(nonce)
        slotted = causal.slot_frozen_case_v1(
            case,
            item.decision_seal,
            target_schedule,
            future_nonce=nonce,
        )
        slotted_cases.append(slotted)
        branch_by_name = _branch_by_name(item)
        branch_rows: list[dict[str, object]] = []
        for position, name in enumerate(causal.policy_order_v1(case.case_index)):
            branch = branch_by_name[name]
            logical_index = case.case_index * len(causal.POLICIES) + position
            query_sha256 = hashlib.sha256(slotted.query.encode("ascii")).hexdigest()
            queries.append(
                future.FutureReleaseQueryV1(
                    logical_index=logical_index,
                    database_path=item.database_path,
                    scope=item.scope,
                    release_id=branch.applied.release_id,
                    query=slotted.query,
                )
            )
            bindings.append(
                _FutureBinding(
                    logical_index=logical_index,
                    case_index=case.case_index,
                    policy=name,
                    application_root_sha256=branch.applied.evidence_root_sha256,
                    database_snapshot_sha256=item.database_sha256,
                    scope_sha256=future._scope_sha256(item.scope),
                    release_id=branch.applied.release_id,
                    release_content_sha256=(branch.applied.release_content_sha256),
                    query_sha256=query_sha256,
                )
            )
            branch_rows.append(
                {
                    "application_root_sha256": branch.applied.evidence_root_sha256,
                    "logical_index": logical_index,
                    "policy": name,
                    "query_sha256": query_sha256,
                    "release_content_sha256": (branch.applied.release_content_sha256),
                    "release_id": branch.applied.release_id,
                    "scope_sha256": future._scope_sha256(item.scope),
                    # The isolated adapter later randomizes actual wire order
                    # by opaque token.  This is only the parent-side paired-row
                    # presentation order, not a carryover-control claim.
                    "logical_position": position,
                }
            )
        schedule_rows.append(
            {
                "branches": branch_rows,
                "case_id": case.case_id,
                "case_index": case.case_index,
                "decision_seal_sha256": item.decision_seal.seal_sha256,
                "database_snapshot_sha256": item.database_sha256,
                "future_nonce_sha256": hashlib.sha256(bytes.fromhex(nonce)).hexdigest(),
                "slotted_case_sha256": slotted.content_sha256,
                "target_category": slotted.target_category,
            }
        )
    if (
        len(queries) != causal.CASE_COUNT * len(causal.POLICIES)
        or len(bindings) != len(queries)
        or len({item.logical_index for item in bindings}) != len(bindings)
    ):
        _fail("run_completeness")
    payload = {
        "beacon_sha256": hashlib.sha256(beacon).hexdigest(),
        "beacon_seal_content_hash": beacon_seal.content_hash,
        "beacon_seal_id": beacon_seal.seal_id,
        "case_count": causal.CASE_COUNT,
        "category_nonce": category_nonce,
        "cases": schedule_rows,
        "complete": True,
        "experiment": _EXPERIMENT,
        "frozen_suite_sha256": suite_sha256,
        "query_count": len(queries),
        "schema_version": _SCHEMA_VERSION,
        "snapshot_seal_content_hash": snapshot_seal.content_hash,
        "snapshot_seal_id": snapshot_seal.seal_id,
        "target_category_schedule_sha256": target_schedule.schedule_sha256,
        "update_seal_content_hash": update_seal.content_hash,
        "update_seal_id": update_seal.seal_id,
    }
    return (
        target_schedule,
        tuple(slotted_cases),
        tuple(queries),
        tuple(bindings),
        payload,
    )


def _join_verified_answers(
    *,
    cases: tuple[causal.FrozenCaseV1, ...],
    captured: tuple[_CapturedCase, ...],
    target_schedule: causal.TargetCategoryScheduleV1,
    slotted_cases: tuple[causal.SlottedCaseV1, ...],
    bindings: tuple[_FutureBinding, ...],
    answers: tuple[future.VerifiedFutureAnswerV1, ...],
) -> tuple[tuple[causal.CausalEvalRowV1, ...], list[dict[str, object]]]:
    if (
        type(answers) is not tuple
        or type(bindings) is not tuple
        or any(type(item) is not future.VerifiedFutureAnswerV1 for item in answers)
        or any(type(item) is not _FutureBinding for item in bindings)
        or len(answers) != len(bindings)
        or tuple(item.logical_index for item in answers)
        != tuple(item.logical_index for item in bindings)
    ):
        _fail("future_completeness")
    answer_by_index = {item.logical_index: item for item in answers}
    binding_by_index = {item.logical_index: item for item in bindings}
    if (
        len(answer_by_index) != len(answers)
        or len(binding_by_index) != len(bindings)
        or len({item.execution_token_sha256 for item in answers}) != len(answers)
        or any(
            not _is_sha256_text(item.execution_token_sha256)
            or not _is_sha256_text(item.response_sha256)
            for item in answers
        )
    ):
        _fail("future_completeness")

    rows: list[causal.CausalEvalRowV1] = []
    # Opaque execution tokens are validated above for this child batch but are
    # deliberately absent from the durable scientific result: transport order
    # is randomized anew on exact retry and the scripted consumer is stateless.
    # A stochastic/model consumer must first seal its execution seed/attempt.
    execution_rows: list[dict[str, object]] = []
    for case, item, slotted in zip(
        cases,
        captured,
        slotted_cases,
        strict=True,
    ):
        branch_by_name = _branch_by_name(item)
        responses: dict[str, str] = {}
        for position, name in enumerate(causal.policy_order_v1(case.case_index)):
            logical_index = case.case_index * len(causal.POLICIES) + position
            binding = binding_by_index[logical_index]
            answer = answer_by_index[logical_index]
            branch = branch_by_name[name]
            expected_query_hash = hashlib.sha256(
                slotted.query.encode("ascii")
            ).hexdigest()
            try:
                observed_response_hash = hashlib.sha256(
                    answer.response.encode("utf-8", errors="strict")
                ).hexdigest()
            except (AttributeError, UnicodeEncodeError) as error:
                _fail("future_binding", error)
            if (
                binding.case_index != case.case_index
                or binding.policy != name
                or binding.release_id != branch.applied.release_id
                or binding.application_root_sha256
                != branch.applied.evidence_root_sha256
                or binding.database_snapshot_sha256 != item.database_sha256
                or binding.scope_sha256 != future._scope_sha256(item.scope)
                or binding.query_sha256 != expected_query_hash
                or answer.release_id != binding.release_id
                or answer.release_content_sha256 != binding.release_content_sha256
                or answer.database_sha256 != binding.database_snapshot_sha256
                or answer.scope_sha256 != binding.scope_sha256
                or answer.query_sha256 != binding.query_sha256
                or answer.response_sha256 != observed_response_hash
                or branch.applied.evidence_root_sha256
                != application.recompute_applied_policy_release_root_v1(branch.applied)
            ):
                _fail("future_binding")
            normalized_response = harness.normalize_response(answer.response)
            responses[name] = normalized_response
            execution_rows.append(
                {
                    "application_root_sha256": binding.application_root_sha256,
                    "database_snapshot_sha256": (binding.database_snapshot_sha256),
                    "logical_index": logical_index,
                    "policy": name,
                    "query_sha256": answer.query_sha256,
                    "release_content_sha256": answer.release_content_sha256,
                    "release_id": answer.release_id,
                    "normalized_response_sha256": hashlib.sha256(
                        normalized_response.encode("utf-8", errors="strict")
                    ).hexdigest(),
                    "response_sha256": answer.response_sha256,
                    "scope_sha256": answer.scope_sha256,
                }
            )
        rows.append(
            causal.make_causal_eval_row_v1(
                case,
                item.decision_seal,
                target_schedule,
                slotted,
                responses,
            )
        )
    return tuple(rows), execution_rows


def _metrics_payload(metrics: causal.CausalEvalMetricsV1) -> dict[str, object]:
    return {
        "aggregate_weighting": metrics.aggregate_weighting,
        "adversarial_stress_categories": list(metrics.adversarial_stress_categories),
        "no_plaintext_root_sha256": metrics.no_plaintext_root_sha256,
        "case_count": metrics.case_count,
        "category_accuracies": [
            {
                "accuracy": item.accuracy,
                "category": item.category,
                "control_role": item.control_role,
                "policy": item.policy,
                "successes": item.successes,
                "total": item.total,
            }
            for item in metrics.category_accuracies
        ],
        "placebo_control_categories": list(metrics.placebo_control_categories),
        "paired_comparisons": [
            {
                "discordant_count": item.discordant_count,
                "exact_sign_test_p_value": item.exact_sign_test_p_value,
                "left_policy": item.left_policy,
                "losses": item.losses,
                "macro_category_delta": item.macro_category_delta,
                "mean_delta": item.mean_delta,
                "right_policy": item.right_policy,
                "ties": item.ties,
                "wins": item.wins,
            }
            for item in metrics.paired_comparisons
        ],
        "policy_accuracies": [
            {
                "accuracy": item.accuracy,
                "policy": item.policy,
                "successes": item.successes,
                "total": item.total,
            }
            for item in metrics.policy_accuracies
        ],
        "schema_version": metrics.schema_version,
    }


def run_local_update_causal_experiment_v1(
    root: str | Path,
    *,
    external_beacon: bytes | None = None,
    future_timeout_seconds: float = 180.0,
) -> LocalUpdateCausalRunReceiptV1:
    """Run exactly 64 paired cases with no attrition or post-hoc replacement."""

    if (
        not isinstance(root, (str, Path))
        or (
            external_beacon is not None
            and (type(external_beacon) is not bytes or len(external_beacon) != 32)
        )
        or type(future_timeout_seconds) not in (int, float)
        or isinstance(future_timeout_seconds, bool)
        or future_timeout_seconds <= 0
        or future_timeout_seconds > 3600
        or (
            type(future_timeout_seconds) is float
            and not math.isfinite(future_timeout_seconds)
        )
    ):
        _fail("closed_schema")
    try:
        run_root = Path(root)
        run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        run_root = _require_private_directory(run_root, reason="root_invalid")
    except LocalUpdateCausalRunError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        _fail("root_invalid", error)

    cases = causal.generate_frozen_cases_v1()
    captured: list[_CapturedCase] = []
    # Nothing after this loop is allowed to run if even one registered case
    # fails; there is no attrition, retry-with-a-new-case, or replacement.
    for case in cases:
        captured.append(
            _capture_case_v1(
                case,
                run_root / "cases" / f"{case.case_index:02d}-{case.case_id}",
            )
        )
    captured_cases = tuple(captured)
    seal_database = run_root / "causal-run-seals.sqlite3"
    update_seal = _seal_and_reopen(
        seal_database,
        idempotency_key=_UPDATE_SEAL_KEY,
        payload=_update_seal_payload(cases, captured_cases),
    )

    # SQLite backup takes a consistent read snapshot.  Parent-side expected
    # treatments and the isolated child then consume these same private,
    # read-only database files rather than live capture databases.
    captured_cases = _freeze_case_databases_v1(run_root, captured_cases)
    snapshot_seal = _seal_and_reopen(
        seal_database,
        idempotency_key=_SNAPSHOT_SEAL_KEY,
        payload=_snapshot_seal_payload(
            captured=captured_cases,
            update_seal=update_seal,
        ),
    )
    # Reopen an existing one-shot beacon on resume, or seal a new one before
    # deriving/exposing any concrete target.
    beacon, _beacon_source, beacon_seal = _resume_or_seal_beacon(
        seal_database,
        snapshot_seal=snapshot_seal,
        update_seal=update_seal,
        external_beacon=external_beacon,
    )
    (
        target_schedule,
        slotted_cases,
        queries,
        bindings,
        schedule_payload,
    ) = _make_future_schedule(
        cases=cases,
        captured=captured_cases,
        update_seal=update_seal,
        snapshot_seal=snapshot_seal,
        beacon_seal=beacon_seal,
        beacon=beacon,
    )
    schedule_seal = _seal_and_reopen(
        seal_database,
        idempotency_key=_SCHEDULE_SEAL_KEY,
        payload=schedule_payload,
    )
    # Drop the in-memory copy before the child is invoked.  The durable beacon
    # seal intentionally retains the public randomness for exact recovery.
    del beacon

    answers = future.execute_verified_release_batch_v1(
        queries,
        timeout_seconds=future_timeout_seconds,
    )
    rows, execution_rows = _join_verified_answers(
        cases=cases,
        captured=captured_cases,
        target_schedule=target_schedule,
        slotted_cases=slotted_cases,
        bindings=bindings,
        answers=answers,
    )
    decision_seals = tuple(item.decision_seal for item in captured_cases)
    metrics = causal.analyze_causal_eval_v1(
        cases,
        decision_seals,
        target_schedule,
        slotted_cases,
        rows,
    )
    result_payload = {
        "complete": True,
        "beacon_seal_content_hash": beacon_seal.content_hash,
        "beacon_seal_id": beacon_seal.seal_id,
        "execution_receipts": execution_rows,
        "experiment": _EXPERIMENT,
        "metrics": _metrics_payload(metrics),
        "response_normalization": "unicode-nfkc-strip-uppercase-v1",
        "result_root_sha256": causal.causal_eval_root_sha256_v1(
            cases,
            decision_seals,
            target_schedule,
            slotted_cases,
            rows,
        ),
        "schedule_seal_content_hash": schedule_seal.content_hash,
        "schedule_seal_id": schedule_seal.seal_id,
        "schema_version": _SCHEMA_VERSION,
        "snapshot_seal_content_hash": snapshot_seal.content_hash,
        "snapshot_seal_id": snapshot_seal.seal_id,
        "target_category_schedule_sha256": target_schedule.schedule_sha256,
        "update_seal_content_hash": update_seal.content_hash,
        "update_seal_id": update_seal.seal_id,
    }
    result_seal = _seal_and_reopen(
        seal_database,
        idempotency_key=_RESULT_SEAL_KEY,
        payload=result_payload,
    )
    return LocalUpdateCausalRunReceiptV1(
        schema_version=_SCHEMA_VERSION,
        experiment=_EXPERIMENT,
        update_seal=update_seal,
        snapshot_seal=snapshot_seal,
        beacon_seal=beacon_seal,
        schedule_seal=schedule_seal,
        result_seal=result_seal,
        target_category_schedule=target_schedule,
        metrics=metrics,
        rows=rows,
    )
