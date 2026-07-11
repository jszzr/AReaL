# SPDX-License-Identifier: Apache-2.0

"""Persist one sealed local-update policy decision into a Memory release.

The function in this module replays the policy input from the exact SQLite
scope and base release before writing.  It then persists evidence-grounded
candidates and ADD/SUPERSEDE revisions and publishes one immutable release as
the visibility barrier.  Retries use decision-derived idempotency keys.

Candidate or revision rows may remain after a crash, but they are not reachable
through any runtime release traversal until the final release exists.  The
builder-produced ``PolicyInputV1`` binds a durable evidence snapshot, including
its global ingestion watermark and complete ordered members.  Later rows,
including backdated rows, belong to a later snapshot and do not invalidate this
one.

This is a crash-conservative single-writer protocol, not an atomic multi-process
transaction, signature, freshness proof, or confidentiality mechanism.  A
trusted writer proves completeness at the selected watermark; orchestration
still decides whether that snapshot is fresh enough.  Hashes and truncated
content IDs can be enumerated when answer spaces are small.  The later
experiment runner must seal every case before it materializes any future query
or outcome and must consume only an explicitly returned receipt/release, never
a release selected by list order or a "latest" heuristic.

Like the policy DTO, this experimental V1 receipt is not a migration target for
an earlier public wire; both shapes land together in the same contribution.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, replace

from examples.memory_service import local_update_policy_eval as policy

from areal.v2.memory_service import (
    CandidateProposal,
    MemoryScope,
    ReleaseManifest,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.errors import MemoryServiceError
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

__all__ = [
    "AppliedPolicyReleaseV1",
    "AppliedUpdateReceiptV1",
    "LocalUpdateApplyError",
    "apply_local_update_decision_v1",
    "recompute_applied_policy_release_root_v1",
]


class LocalUpdateApplyError(RuntimeError):
    """Stable, answer-free reason for refusing or failing publication."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("apply error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class AppliedUpdateReceiptV1:
    ordinal: int
    release_position: int
    operation: str
    evidence_count: int
    update_commitment_sha256: str
    candidate_id: str
    candidate_content_sha256: str
    revision_id: str
    revision_content_sha256: str
    memory_id: str
    generation: int
    parent_revision_id: str | None


@dataclass(frozen=True, slots=True)
class AppliedPolicyReleaseV1:
    """Answer-preimage-free receipt for one published policy decision."""

    schema_version: int
    seal_policy: str
    application_id: str
    policy: str
    policy_scope_token: str
    input_sha256: str
    decision_sha256: str
    base_release_id: str
    base_release_content_sha256: str
    release_id: str
    release_content_sha256: str
    changed: bool
    base_revision_count: int
    result_revision_count: int
    update_count: int
    revision_ids: tuple[str, ...]
    updates: tuple[AppliedUpdateReceiptV1, ...]
    evidence_root_sha256: str


_SCHEMA_VERSION = 1
_SEAL_POLICY = "same-scope-durable-snapshot-revalidated-release-last-v1"
_APPLICATION_ID_DOMAIN = b"areal-memory-local-update-application-id-v1\0"
_UPDATE_HASH_DOMAIN = b"areal-memory-local-update-application-v1\0"
_ROOT_DOMAIN = b"areal-memory-local-update-release-evidence-v1\0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_APPLICATION_ID_PATTERN = re.compile(r"apply_[0-9a-f]{64}")
_SCOPE_TOKEN_PATTERN = re.compile(r"scope_[0-9a-f]{64}")
_CANDIDATE_ID_PATTERN = re.compile(r"cand_[0-9a-f]{24}")
_REVISION_ID_PATTERN = re.compile(r"rev_[0-9a-f]{24}")
_MEMORY_ID_PATTERN = re.compile(r"mem_[0-9a-f]{24}")
_RELEASE_ID_PATTERN = re.compile(r"rel_[0-9a-f]{24}")


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
        raise LocalUpdateApplyError("receipt_invalid") from error


def _update_commitment(update: policy.PolicyUpdateV1) -> str:
    value = {
        "content": update.content,
        "evidence_ids": list(update.evidence_ids),
        "key": update.key,
        "operation": update.operation,
        "parent_revision_id": update.parent_revision_id,
        "value": update.value,
    }
    return hashlib.sha256(
        _UPDATE_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _receipt_value(value: AppliedUpdateReceiptV1) -> dict[str, object]:
    if (
        type(value) is not AppliedUpdateReceiptV1
        or type(value.ordinal) is not int
        or value.ordinal < 0
        or type(value.release_position) is not int
        or value.release_position < 0
        or type(value.operation) is not str
        or value.operation not in {"add", "supersede"}
        or type(value.evidence_count) is not int
        or value.evidence_count <= 0
        or type(value.update_commitment_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.update_commitment_sha256) is None
        or type(value.candidate_id) is not str
        or _CANDIDATE_ID_PATTERN.fullmatch(value.candidate_id) is None
        or type(value.candidate_content_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.candidate_content_sha256) is None
        or value.candidate_id != f"cand_{value.candidate_content_sha256[:24]}"
        or type(value.revision_id) is not str
        or _REVISION_ID_PATTERN.fullmatch(value.revision_id) is None
        or type(value.revision_content_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.revision_content_sha256) is None
        or value.revision_id != f"rev_{value.revision_content_sha256[:24]}"
        or type(value.memory_id) is not str
        or _MEMORY_ID_PATTERN.fullmatch(value.memory_id) is None
        or type(value.generation) is not int
        or value.generation < 0
        or value.generation > 2**63 - 1
        or (
            value.operation == "add"
            and (
                value.generation != 0
                or value.parent_revision_id is not None
                or value.memory_id != f"mem_{value.revision_content_sha256[:24]}"
            )
        )
        or (
            value.operation == "supersede"
            and (
                value.generation <= 0
                or type(value.parent_revision_id) is not str
                or _REVISION_ID_PATTERN.fullmatch(value.parent_revision_id) is None
            )
        )
    ):
        raise LocalUpdateApplyError("receipt_invalid")
    return {
        "candidate_content_sha256": value.candidate_content_sha256,
        "candidate_id": value.candidate_id,
        "evidence_count": value.evidence_count,
        "generation": value.generation,
        "memory_id": value.memory_id,
        "operation": value.operation,
        "ordinal": value.ordinal,
        "release_position": value.release_position,
        "parent_revision_id": value.parent_revision_id,
        "revision_content_sha256": value.revision_content_sha256,
        "revision_id": value.revision_id,
        "update_commitment_sha256": value.update_commitment_sha256,
    }


def _root_value(value: AppliedPolicyReleaseV1) -> dict[str, object]:
    if (
        type(value) is not AppliedPolicyReleaseV1
        or type(value.schema_version) is not int
        or value.schema_version != _SCHEMA_VERSION
        or type(value.seal_policy) is not str
        or value.seal_policy != _SEAL_POLICY
        or type(value.application_id) is not str
        or _APPLICATION_ID_PATTERN.fullmatch(value.application_id) is None
        or type(value.policy) is not str
        or value.policy not in policy.SUPPORTED_POLICIES
        or type(value.policy_scope_token) is not str
        or _SCOPE_TOKEN_PATTERN.fullmatch(value.policy_scope_token) is None
        or type(value.input_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.input_sha256) is None
        or type(value.decision_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.decision_sha256) is None
        or type(value.base_release_id) is not str
        or _RELEASE_ID_PATTERN.fullmatch(value.base_release_id) is None
        or type(value.base_release_content_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.base_release_content_sha256) is None
        or value.base_release_id != f"rel_{value.base_release_content_sha256[:24]}"
        or type(value.release_id) is not str
        or _RELEASE_ID_PATTERN.fullmatch(value.release_id) is None
        or type(value.release_content_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.release_content_sha256) is None
        or value.release_id != f"rel_{value.release_content_sha256[:24]}"
        or type(value.changed) is not bool
        or type(value.base_revision_count) is not int
        or value.base_revision_count < 0
        or type(value.result_revision_count) is not int
        or value.result_revision_count < 0
        or type(value.update_count) is not int
        or type(value.revision_ids) is not tuple
        or any(
            type(item) is not str or _REVISION_ID_PATTERN.fullmatch(item) is None
            for item in value.revision_ids
        )
        or type(value.updates) is not tuple
        or any(type(item) is not AppliedUpdateReceiptV1 for item in value.updates)
        or value.update_count != len(value.updates)
        or tuple(item.ordinal for item in value.updates)
        != tuple(range(value.update_count))
        or len(set(value.revision_ids)) != len(value.revision_ids)
        or value.result_revision_count != len(value.revision_ids)
        or value.result_revision_count
        != value.base_revision_count
        + sum(item.operation == "add" for item in value.updates)
        or len({item.release_position for item in value.updates}) != len(value.updates)
        or any(
            item.release_position >= value.result_revision_count
            or value.revision_ids[item.release_position] != item.revision_id
            for item in value.updates
        )
        or (value.changed is not bool(value.updates))
        or (
            not value.changed
            and (
                value.release_id != value.base_release_id
                or value.release_content_sha256 != value.base_release_content_sha256
            )
        )
        or (
            value.changed
            and (
                value.release_id == value.base_release_id
                or value.release_content_sha256 == value.base_release_content_sha256
            )
        )
    ):
        raise LocalUpdateApplyError("receipt_invalid")
    application_value = {
        "base_release_content_sha256": value.base_release_content_sha256,
        "base_release_id": value.base_release_id,
        "decision_sha256": value.decision_sha256,
        "input_sha256": value.input_sha256,
        "policy": value.policy,
        "policy_scope_token": value.policy_scope_token,
    }
    expected_application_id = (
        "apply_"
        + hashlib.sha256(
            _APPLICATION_ID_DOMAIN + _canonical_json_bytes(application_value)
        ).hexdigest()
    )
    if value.application_id != expected_application_id:
        raise LocalUpdateApplyError("receipt_invalid")
    return {
        "application_id": value.application_id,
        "base_revision_count": value.base_revision_count,
        "base_release_content_sha256": value.base_release_content_sha256,
        "base_release_id": value.base_release_id,
        "changed": value.changed,
        "decision_sha256": value.decision_sha256,
        "input_sha256": value.input_sha256,
        "policy": value.policy,
        "policy_scope_token": value.policy_scope_token,
        "release_content_sha256": value.release_content_sha256,
        "release_id": value.release_id,
        "revision_ids": list(value.revision_ids),
        "schema_version": value.schema_version,
        "seal_policy": value.seal_policy,
        "result_revision_count": value.result_revision_count,
        "update_count": value.update_count,
        "updates": [_receipt_value(item) for item in value.updates],
    }


def recompute_applied_policy_release_root_v1(
    value: AppliedPolicyReleaseV1,
) -> str:
    """Recompute the domain-separated root without trusting the stored root."""

    return hashlib.sha256(
        _ROOT_DOMAIN + _canonical_json_bytes(_root_value(value))
    ).hexdigest()


def _replay_policy_input(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    value: policy.PolicyInputV1,
) -> policy.PolicyInputV1:
    try:
        policy.policy_input_wire_v1(value)
    except (policy.LocalUpdatePolicyError, TypeError, ValueError, OverflowError):
        raise LocalUpdateApplyError("input_invalid") from None
    try:
        live_projection = policy.project_policy_input_from_snapshot_v1(
            store=store,
            scope=scope,
            base_release_id=value.base_release_id,
            evidence_snapshot_id=value.evidence_snapshot_id,
        )
    except (policy.LocalUpdatePolicyError, TypeError, ValueError, OverflowError):
        raise LocalUpdateApplyError("input_drift") from None
    if live_projection != value:
        raise LocalUpdateApplyError("input_drift")
    return value


def _validate_decision(
    *,
    value: policy.PolicyInputV1,
    decision: policy.PolicyDecisionV1,
    expected_policy: str,
) -> str:
    try:
        policy.validate_local_update_decision_v1(
            value,
            decision,
            expected_policy=expected_policy,
        )
        return policy.policy_decision_sha256_v1(decision)
    except policy.LocalUpdatePolicyError:
        raise LocalUpdateApplyError("decision_invalid") from None


def _application_id(
    *,
    policy_input: policy.PolicyInputV1,
    expected_policy: str,
    input_sha256: str,
    decision_sha256: str,
) -> str:
    value = {
        "base_release_content_sha256": policy_input.base_release_content_sha256,
        "base_release_id": policy_input.base_release_id,
        "decision_sha256": decision_sha256,
        "input_sha256": input_sha256,
        "policy": expected_policy,
        "policy_scope_token": policy_input.policy_scope_token,
    }
    return (
        "apply_"
        + hashlib.sha256(
            _APPLICATION_ID_DOMAIN + _canonical_json_bytes(value)
        ).hexdigest()
    )


def apply_local_update_decision_v1(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    policy_input: policy.PolicyInputV1,
    decision: policy.PolicyDecisionV1,
    expected_policy: str,
) -> AppliedPolicyReleaseV1:
    """Replay, persist, publish, and return one answer-free application receipt."""

    if type(store) is not SQLiteMemoryStore or type(scope) is not MemoryScope:
        raise LocalUpdateApplyError("input_invalid")
    replayed = _replay_policy_input(store=store, scope=scope, value=policy_input)
    decision_sha256 = _validate_decision(
        value=replayed,
        decision=decision,
        expected_policy=expected_policy,
    )
    input_sha256 = policy.policy_input_sha256_v1(replayed)
    application_id = _application_id(
        policy_input=replayed,
        expected_policy=expected_policy,
        input_sha256=input_sha256,
        decision_sha256=decision_sha256,
    )
    base_by_key = {item.key: item for item in replayed.base_memories}
    expected_base_by_key = dict(base_by_key)
    try:
        base_release = store.get_release(scope, replayed.base_release_id)
        base_revisions = store.get_release_revisions(
            scope,
            replayed.base_release_id,
        )
    except (MemoryServiceError, sqlite3.Error, OSError):
        raise LocalUpdateApplyError("input_drift") from None
    base_revision_ids = tuple(item.revision_id for item in base_revisions)
    if (
        base_release.content_hash != replayed.base_release_content_sha256
        or base_release.manifest.revision_ids != base_revision_ids
        or {item.revision_id for item in replayed.base_memories}
        != set(base_revision_ids)
    ):
        raise LocalUpdateApplyError("input_drift")

    # Keep the final fail-closed read adjacent to the first candidate write.
    # Public snapshots are immutable; this guard also catches injected storage
    # drift before any decision-derived orphan can be created.
    _replay_policy_input(store=store, scope=scope, value=replayed)
    result_revision_ids = list(base_revision_ids)
    receipts: list[AppliedUpdateReceiptV1] = []

    for ordinal, update in enumerate(decision.updates):
        prefix = f"{application_id}-{ordinal:03d}"
        try:
            candidate_proposal = CandidateProposal(
                scope=scope,
                content=update.content,
                evidence_ids=update.evidence_ids,
                idempotency_key=f"{prefix}-candidate",
            )
            candidate = store.append_candidate(candidate_proposal)
            grounding = store.get_candidate_evidence(scope, candidate.candidate_id)
            revision_proposal = RevisionProposal(
                scope=scope,
                candidate_id=candidate.candidate_id,
                operation=(
                    RevisionOperation.ADD
                    if update.operation == "add"
                    else RevisionOperation.SUPERSEDE
                ),
                parent_revision_id=update.parent_revision_id,
                idempotency_key=f"{prefix}-revision",
            )
            revision = store.append_revision(revision_proposal)
        except (MemoryServiceError, sqlite3.Error, OSError):
            raise LocalUpdateApplyError("persistence_invalid") from None

        base = base_by_key.get(update.key)
        if (
            candidate.proposal != candidate_proposal
            or candidate.proposal.evidence_ids
            != tuple(item.evidence_id for item in grounding)
            or any(item.event.scope != scope for item in grounding)
            or revision.proposal != revision_proposal
            or (
                update.operation == "add"
                and (base is not None or revision.generation != 0)
            )
            or (
                update.operation == "supersede"
                and (
                    base is None
                    or revision.memory_id != base.memory_id
                    or revision.generation != base.generation + 1
                )
            )
        ):
            raise LocalUpdateApplyError("persistence_invalid")
        if update.operation == "supersede":
            assert update.parent_revision_id is not None
            try:
                release_position = result_revision_ids.index(update.parent_revision_id)
            except ValueError:
                raise LocalUpdateApplyError("persistence_invalid") from None
            result_revision_ids[release_position] = revision.revision_id
        else:
            release_position = len(result_revision_ids)
            result_revision_ids.append(revision.revision_id)
        expected_base_by_key[update.key] = policy.BaseMemoryV1(
            key=update.key,
            value=update.value,
            memory_id=revision.memory_id,
            revision_id=revision.revision_id,
            generation=revision.generation,
        )
        receipts.append(
            AppliedUpdateReceiptV1(
                ordinal=ordinal,
                release_position=release_position,
                operation=update.operation,
                evidence_count=len(update.evidence_ids),
                update_commitment_sha256=_update_commitment(update),
                candidate_id=candidate.candidate_id,
                candidate_content_sha256=candidate.content_hash,
                revision_id=revision.revision_id,
                revision_content_sha256=revision.content_hash,
                memory_id=revision.memory_id,
                generation=revision.generation,
                parent_revision_id=revision.proposal.parent_revision_id,
            )
        )

    # Revalidate every sealed member before crossing the immutable release
    # publication barrier.  Later evidence is outside this frozen snapshot.
    _replay_policy_input(store=store, scope=scope, value=replayed)
    try:
        manifest = ReleaseManifest(
            scope=scope,
            revision_ids=tuple(result_revision_ids),
        )
        release = store.append_release(
            manifest,
            idempotency_key=f"{application_id}-release",
        )
        revisions = store.get_release_revisions(scope, release.release_id)
        post_projection = policy.project_policy_input_from_snapshot_v1(
            store=store,
            scope=scope,
            base_release_id=release.release_id,
            evidence_snapshot_id=replayed.evidence_snapshot_id,
        )
    except (
        MemoryServiceError,
        sqlite3.Error,
        OSError,
        policy.LocalUpdatePolicyError,
    ):
        raise LocalUpdateApplyError("publication_invalid") from None

    revision_ids = tuple(item.revision_id for item in revisions)
    expected_revision_ids = tuple(result_revision_ids)
    expected_base_memories = tuple(
        sorted(expected_base_by_key.values(), key=lambda item: item.key)
    )
    expected_post_projection = replace(
        replayed,
        base_release_id=release.release_id,
        base_release_content_sha256=release.content_hash,
        base_memories=expected_base_memories,
    )
    if (
        release.manifest.scope != scope
        or release.manifest.revision_ids != revision_ids
        or revision_ids != expected_revision_ids
        or (bool(receipts) and release.release_id == replayed.base_release_id)
        or (not receipts and release.release_id != replayed.base_release_id)
        or post_projection != expected_post_projection
    ):
        raise LocalUpdateApplyError("publication_invalid")

    provisional = AppliedPolicyReleaseV1(
        schema_version=_SCHEMA_VERSION,
        seal_policy=_SEAL_POLICY,
        application_id=application_id,
        policy=decision.policy,
        policy_scope_token=replayed.policy_scope_token,
        input_sha256=input_sha256,
        decision_sha256=decision_sha256,
        base_release_id=replayed.base_release_id,
        base_release_content_sha256=replayed.base_release_content_sha256,
        release_id=release.release_id,
        release_content_sha256=release.content_hash,
        changed=bool(receipts),
        base_revision_count=len(base_revision_ids),
        result_revision_count=len(revision_ids),
        update_count=len(receipts),
        revision_ids=revision_ids,
        updates=tuple(receipts),
        evidence_root_sha256="",
    )
    root = recompute_applied_policy_release_root_v1(provisional)
    return AppliedPolicyReleaseV1(
        schema_version=provisional.schema_version,
        seal_policy=provisional.seal_policy,
        application_id=provisional.application_id,
        policy=provisional.policy,
        policy_scope_token=provisional.policy_scope_token,
        input_sha256=provisional.input_sha256,
        decision_sha256=provisional.decision_sha256,
        base_release_id=provisional.base_release_id,
        base_release_content_sha256=provisional.base_release_content_sha256,
        release_id=provisional.release_id,
        release_content_sha256=provisional.release_content_sha256,
        changed=provisional.changed,
        base_revision_count=provisional.base_revision_count,
        result_revision_count=provisional.result_revision_count,
        update_count=provisional.update_count,
        revision_ids=provisional.revision_ids,
        updates=provisional.updates,
        evidence_root_sha256=root,
    )
