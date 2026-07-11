# SPDX-License-Identifier: Apache-2.0

"""Closed, deterministic policy kernel for local Memory update experiments.

The policy receives only an opaque scope token, a cutoff-bounded projection of
persisted evidence, and the currently released base memories.  It does not
receive a case index, future query, arm label, scorer truth, reward, utility,
outcome, database path, or store capability.

This module deliberately stops before persistence and scoring.  A later runner
can turn a validated decision into candidates, revisions, and a release, then
measure it on future tasks.  Keeping the policy boundary pure makes that causal
experiment auditable: a behavioral gain cannot be attributed to an accidental
truth or reward callback hidden in the update API.

The boundary is an honest-code dataflow contract, not a malicious Python
sandbox.  A policy that imports experiment internals or exploits global state is
outside the claim made here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from areal.v2.memory_service import EvidenceKind, MemoryScope
from areal.v2.memory_service.errors import MemoryServiceError
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

__all__ = [
    "BaseMemoryV1",
    "LocalUpdatePolicyError",
    "PolicyDecisionV1",
    "PolicyEvidenceV1",
    "PolicyInputV1",
    "PolicyUpdateV1",
    "POLICY_EVIDENCE_KINDS",
    "SUPPORTED_POLICIES",
    "make_policy_input_v1",
    "policy_decision_sha256_v1",
    "policy_decision_wire_v1",
    "policy_input_sha256_v1",
    "policy_input_wire_v1",
    "run_local_update_policy_v1",
    "validate_local_update_decision_v1",
]


class LocalUpdatePolicyError(ValueError):
    """Stable, answer-free reason for rejecting a policy input or decision."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("policy error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class PolicyEvidenceV1:
    evidence_id: str
    kind: str
    payload: str
    observed_at_utc: str
    sequence_no: int


@dataclass(frozen=True, slots=True)
class BaseMemoryV1:
    key: str
    value: str
    memory_id: str
    revision_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class PolicyInputV1:
    """The complete input visible to one update-policy invocation.

    This DTO is constructible and is not a store-authenticity capability.  The
    experiment runner must obtain it from :func:`make_policy_input_v1` and bind
    its hash before any future query or outcome exists.
    """

    schema_version: int
    policy_scope_token: str
    base_release_id: str
    base_release_content_sha256: str
    cutoff_utc: str
    evidence: tuple[PolicyEvidenceV1, ...]
    base_memories: tuple[BaseMemoryV1, ...]


@dataclass(frozen=True, slots=True)
class PolicyUpdateV1:
    key: str
    value: str
    content: str
    evidence_ids: tuple[str, ...]
    operation: str
    parent_revision_id: str | None


@dataclass(frozen=True, slots=True)
class PolicyDecisionV1:
    schema_version: int
    policy: str
    input_sha256: str
    updates: tuple[PolicyUpdateV1, ...]


_SCHEMA_VERSION = 1
_INPUT_HASH_DOMAIN = b"areal-memory-local-update-policy-input-v1\0"
_DECISION_HASH_DOMAIN = b"areal-memory-local-update-policy-decision-v1\0"
_SCOPE_TOKEN_DOMAIN = b"areal-memory-local-update-policy-scope-v1\0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_PATTERN = re.compile(r"evd_[0-9a-f]{24}")
_MEMORY_ID_PATTERN = re.compile(r"mem_[0-9a-f]{24}")
_REVISION_ID_PATTERN = re.compile(r"rev_[0-9a-f]{24}")
_RELEASE_ID_PATTERN = re.compile(r"rel_[0-9a-f]{24}")
_SCOPE_TOKEN_PATTERN = re.compile(r"scope_[0-9a-f]{64}")
_KEY_PATTERN = re.compile(r"project-[abcdefghjklmnpqrstuvwxyz23456789]{6}")
_VALUE_PATTERN = re.compile(r"[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5}")
_FACT_PATTERN = re.compile(
    r"(?P<key>project-[abcdefghjklmnpqrstuvwxyz23456789]{6}) = "
    r"(?P<value>[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5})"
)

SUPPORTED_POLICIES = ("feedback_latest", "noop", "latest_any")
POLICY_EVIDENCE_KINDS = (
    EvidenceKind.USER_MESSAGE.value,
    EvidenceKind.FEEDBACK.value,
)


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
        raise LocalUpdatePolicyError("closed_schema") from error


def _canonical_utc_text(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is None:
        raise LocalUpdatePolicyError("closed_schema")
    try:
        normalized = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise LocalUpdatePolicyError("closed_schema") from error
    return normalized.isoformat()


def _parse_canonical_utc_text(value: object) -> datetime:
    if type(value) is not str:
        raise LocalUpdatePolicyError("closed_schema")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise LocalUpdatePolicyError("closed_schema") from error
    if parsed.tzinfo is None or _canonical_utc_text(parsed) != value:
        raise LocalUpdatePolicyError("closed_schema")
    return parsed


def _parse_fact(payload: object) -> tuple[str, str] | None:
    if type(payload) is not str:
        raise LocalUpdatePolicyError("closed_schema")
    match = _FACT_PATTERN.fullmatch(payload)
    if match is None:
        return None
    return match.group("key"), match.group("value")


def _scope_token(scope: MemoryScope) -> str:
    if type(scope) is not MemoryScope:
        raise LocalUpdatePolicyError("closed_schema")
    value = {
        "namespace": scope.namespace,
        "subject_id": scope.subject_id,
        "tenant_id": scope.tenant_id,
    }
    digest = hashlib.sha256(
        _SCOPE_TOKEN_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()
    return f"scope_{digest}"


def _evidence_sort_key(
    evidence: PolicyEvidenceV1,
) -> tuple[datetime, int, str]:
    # Pre-registered total order: later tuple wins per key.  The content-derived
    # evidence ID is the deterministic final tie-break when time and sequence tie.
    return (
        _parse_canonical_utc_text(evidence.observed_at_utc),
        evidence.sequence_no,
        evidence.evidence_id,
    )


def _validate_policy_evidence(value: object) -> PolicyEvidenceV1:
    if type(value) is not PolicyEvidenceV1:
        raise LocalUpdatePolicyError("closed_schema")
    if (
        type(value.evidence_id) is not str
        or _EVIDENCE_ID_PATTERN.fullmatch(value.evidence_id) is None
        or type(value.kind) is not str
        or value.kind not in POLICY_EVIDENCE_KINDS
        or type(value.payload) is not str
        or type(value.sequence_no) is not int
        or value.sequence_no < 0
        or value.sequence_no > 2**63 - 1
    ):
        raise LocalUpdatePolicyError("closed_schema")
    try:
        value.payload.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise LocalUpdatePolicyError("closed_schema") from error
    _parse_canonical_utc_text(value.observed_at_utc)
    return value


def _validate_base_memory(value: object) -> BaseMemoryV1:
    if type(value) is not BaseMemoryV1:
        raise LocalUpdatePolicyError("closed_schema")
    if (
        type(value.key) is not str
        or _KEY_PATTERN.fullmatch(value.key) is None
        or type(value.value) is not str
        or _VALUE_PATTERN.fullmatch(value.value) is None
        or type(value.memory_id) is not str
        or _MEMORY_ID_PATTERN.fullmatch(value.memory_id) is None
        or type(value.revision_id) is not str
        or _REVISION_ID_PATTERN.fullmatch(value.revision_id) is None
        or type(value.generation) is not int
        or value.generation < 0
        or value.generation > 2**63 - 1
    ):
        raise LocalUpdatePolicyError("closed_schema")
    return value


def _validate_policy_input(value: object) -> PolicyInputV1:
    if (
        type(value) is not PolicyInputV1
        or type(value.schema_version) is not int
        or value.schema_version != _SCHEMA_VERSION
        or type(value.policy_scope_token) is not str
        or _SCOPE_TOKEN_PATTERN.fullmatch(value.policy_scope_token) is None
        or type(value.base_release_id) is not str
        or _RELEASE_ID_PATTERN.fullmatch(value.base_release_id) is None
        or type(value.base_release_content_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.base_release_content_sha256) is None
        or type(value.evidence) is not tuple
        or type(value.base_memories) is not tuple
    ):
        raise LocalUpdatePolicyError("closed_schema")
    cutoff = _parse_canonical_utc_text(value.cutoff_utc)
    evidence = tuple(_validate_policy_evidence(item) for item in value.evidence)
    bases = tuple(_validate_base_memory(item) for item in value.base_memories)
    if (
        evidence != tuple(sorted(evidence, key=_evidence_sort_key))
        or len({item.evidence_id for item in evidence}) != len(evidence)
        or any(
            _parse_canonical_utc_text(item.observed_at_utc) > cutoff
            for item in evidence
        )
        or bases != tuple(sorted(bases, key=lambda item: item.key))
        or len({item.key for item in bases}) != len(bases)
        or len({item.memory_id for item in bases}) != len(bases)
        or len({item.revision_id for item in bases}) != len(bases)
    ):
        raise LocalUpdatePolicyError("input_invariant")
    return value


def _policy_input_value(value: PolicyInputV1) -> dict[str, object]:
    _validate_policy_input(value)
    return {
        "base_release_content_sha256": value.base_release_content_sha256,
        "base_release_id": value.base_release_id,
        "base_memories": [
            {
                "generation": item.generation,
                "key": item.key,
                "memory_id": item.memory_id,
                "revision_id": item.revision_id,
                "value": item.value,
            }
            for item in value.base_memories
        ],
        "cutoff_utc": value.cutoff_utc,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "kind": item.kind,
                "observed_at_utc": item.observed_at_utc,
                "payload": item.payload,
                "sequence_no": item.sequence_no,
            }
            for item in value.evidence
        ],
        "policy_scope_token": value.policy_scope_token,
        "schema_version": value.schema_version,
    }


def policy_input_wire_v1(value: PolicyInputV1) -> bytes:
    """Return the exact compact ASCII JSON supplied to a policy boundary."""

    return _canonical_json_bytes(_policy_input_value(value))


def policy_input_sha256_v1(value: PolicyInputV1) -> str:
    return hashlib.sha256(_INPUT_HASH_DOMAIN + policy_input_wire_v1(value)).hexdigest()


def make_policy_input_v1(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    base_release_id: str,
    cutoff: datetime,
) -> PolicyInputV1:
    """Project one same-scope release and pre-evaluation evidence snapshot."""

    if (
        type(store) is not SQLiteMemoryStore
        or type(scope) is not MemoryScope
        or type(base_release_id) is not str
        or _RELEASE_ID_PATTERN.fullmatch(base_release_id) is None
    ):
        raise LocalUpdatePolicyError("closed_schema")
    cutoff_text = _canonical_utc_text(cutoff)
    try:
        release = store.get_release(scope, base_release_id)
        revisions = store.get_release_revisions(scope, base_release_id)
        if (
            release.manifest.scope != scope
            or release.release_id != base_release_id
            or tuple(item.revision_id for item in revisions)
            != release.manifest.revision_ids
        ):
            raise LocalUpdatePolicyError("base_release_invalid")
        base_memories: list[BaseMemoryV1] = []
        for revision in revisions:
            candidate = store.get_candidate(scope, revision.proposal.candidate_id)
            evidence_records = store.get_candidate_evidence(
                scope, candidate.candidate_id
            )
            parsed = _parse_fact(candidate.proposal.content)
            if (
                parsed is None
                or revision.proposal.scope != scope
                or candidate.proposal.scope != scope
                or revision.proposal.candidate_id != candidate.candidate_id
                or candidate.proposal.evidence_ids
                != tuple(item.evidence_id for item in evidence_records)
                or not evidence_records
                or any(item.event.scope != scope for item in evidence_records)
                or any(
                    _parse_fact(item.event.payload) != parsed
                    for item in evidence_records
                )
            ):
                raise LocalUpdatePolicyError("base_release_invalid")
            key, fact_value = parsed
            base_memories.append(
                BaseMemoryV1(
                    key=key,
                    value=fact_value,
                    memory_id=revision.memory_id,
                    revision_id=revision.revision_id,
                    generation=revision.generation,
                )
            )
        records = tuple(
            record
            for record in store.list(scope)
            if record.event.observed_at <= cutoff.astimezone(UTC)
            and record.event.kind.value in POLICY_EVIDENCE_KINDS
        )
    except LocalUpdatePolicyError:
        raise
    except (MemoryServiceError, TypeError, ValueError, OverflowError) as error:
        raise LocalUpdatePolicyError("base_release_invalid") from error

    projected = tuple(
        PolicyEvidenceV1(
            evidence_id=record.evidence_id,
            kind=record.event.kind.value,
            payload=record.event.payload,
            observed_at_utc=_canonical_utc_text(record.event.observed_at),
            sequence_no=record.event.sequence_no,
        )
        for record in records
    )
    result = PolicyInputV1(
        schema_version=_SCHEMA_VERSION,
        policy_scope_token=_scope_token(scope),
        base_release_id=release.release_id,
        base_release_content_sha256=release.content_hash,
        cutoff_utc=cutoff_text,
        evidence=tuple(sorted(projected, key=_evidence_sort_key)),
        base_memories=tuple(sorted(base_memories, key=lambda item: item.key)),
    )
    return _validate_policy_input(result)


def _selected_facts(
    policy: str,
    value: PolicyInputV1,
) -> tuple[tuple[str, str, str], ...]:
    if policy == "noop":
        return ()
    selected: dict[str, tuple[str, str]] = {}
    for evidence in value.evidence:
        if policy == "feedback_latest" and evidence.kind != EvidenceKind.FEEDBACK.value:
            continue
        parsed = _parse_fact(evidence.payload)
        if parsed is None:
            continue
        key, fact_value = parsed
        selected[key] = (fact_value, evidence.evidence_id)
    return tuple(
        (key, fact_value, evidence_id)
        for key, (fact_value, evidence_id) in sorted(selected.items())
    )


def _expected_decision(policy: str, value: PolicyInputV1) -> PolicyDecisionV1:
    if type(policy) is not str or policy not in SUPPORTED_POLICIES:
        raise LocalUpdatePolicyError("unsupported_policy")
    validated = _validate_policy_input(value)
    base_by_key = {item.key: item for item in validated.base_memories}
    updates: list[PolicyUpdateV1] = []
    for key, fact_value, evidence_id in _selected_facts(policy, validated):
        base = base_by_key.get(key)
        if base is not None and base.value == fact_value:
            continue
        updates.append(
            PolicyUpdateV1(
                key=key,
                value=fact_value,
                content=f"{key} = {fact_value}",
                evidence_ids=(evidence_id,),
                operation="add" if base is None else "supersede",
                parent_revision_id=None if base is None else base.revision_id,
            )
        )
    return PolicyDecisionV1(
        schema_version=_SCHEMA_VERSION,
        policy=policy,
        input_sha256=policy_input_sha256_v1(validated),
        updates=tuple(updates),
    )


def run_local_update_policy_v1(
    policy: str,
    value: PolicyInputV1,
) -> PolicyDecisionV1:
    """Run one fixed, answer-independent policy exactly once."""

    decision = _expected_decision(policy, value)
    return validate_local_update_decision_v1(
        value,
        decision,
        expected_policy=policy,
    )


def _policy_decision_value(value: PolicyDecisionV1) -> dict[str, object]:
    if (
        type(value) is not PolicyDecisionV1
        or type(value.schema_version) is not int
        or value.schema_version != _SCHEMA_VERSION
        or type(value.policy) is not str
        or value.policy not in SUPPORTED_POLICIES
        or type(value.input_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.input_sha256) is None
        or type(value.updates) is not tuple
    ):
        raise LocalUpdatePolicyError("closed_schema")
    updates: list[dict[str, object]] = []
    previous_key: str | None = None
    for update in value.updates:
        if (
            type(update) is not PolicyUpdateV1
            or type(update.key) is not str
            or _KEY_PATTERN.fullmatch(update.key) is None
            or type(update.value) is not str
            or _VALUE_PATTERN.fullmatch(update.value) is None
            or type(update.content) is not str
            or update.content != f"{update.key} = {update.value}"
            or type(update.evidence_ids) is not tuple
            or len(update.evidence_ids) != 1
            or type(update.evidence_ids[0]) is not str
            or _EVIDENCE_ID_PATTERN.fullmatch(update.evidence_ids[0]) is None
            or type(update.operation) is not str
            or update.operation not in {"add", "supersede"}
            or (update.operation == "add" and update.parent_revision_id is not None)
            or (
                update.operation == "supersede"
                and (
                    type(update.parent_revision_id) is not str
                    or _REVISION_ID_PATTERN.fullmatch(update.parent_revision_id) is None
                )
            )
            or (previous_key is not None and update.key <= previous_key)
        ):
            raise LocalUpdatePolicyError("decision_invariant")
        previous_key = update.key
        updates.append(
            {
                "content": update.content,
                "evidence_ids": list(update.evidence_ids),
                "key": update.key,
                "operation": update.operation,
                "parent_revision_id": update.parent_revision_id,
                "value": update.value,
            }
        )
    return {
        "input_sha256": value.input_sha256,
        "policy": value.policy,
        "schema_version": value.schema_version,
        "updates": updates,
    }


def policy_decision_wire_v1(value: PolicyDecisionV1) -> bytes:
    return _canonical_json_bytes(_policy_decision_value(value))


def policy_decision_sha256_v1(value: PolicyDecisionV1) -> str:
    return hashlib.sha256(
        _DECISION_HASH_DOMAIN + policy_decision_wire_v1(value)
    ).hexdigest()


def validate_local_update_decision_v1(
    policy_input: PolicyInputV1,
    decision: PolicyDecisionV1,
    *,
    expected_policy: str,
) -> PolicyDecisionV1:
    """Recompute policy semantics and exact evidence grounding fail-closed."""

    validated_input = _validate_policy_input(policy_input)
    _policy_decision_value(decision)
    if (
        type(expected_policy) is not str
        or expected_policy not in SUPPORTED_POLICIES
        or decision.policy != expected_policy
    ):
        raise LocalUpdatePolicyError("policy_mismatch")
    expected = _expected_decision(expected_policy, validated_input)
    if decision != expected:
        raise LocalUpdatePolicyError("decision_mismatch")

    evidence_by_id = {item.evidence_id: item for item in validated_input.evidence}
    base_by_key = {item.key: item for item in validated_input.base_memories}
    for update in decision.updates:
        evidence = evidence_by_id.get(update.evidence_ids[0])
        base = base_by_key.get(update.key)
        if (
            evidence is None
            or _parse_fact(evidence.payload) != (update.key, update.value)
            or (
                update.operation == "add"
                and (base is not None or update.parent_revision_id is not None)
            )
            or (
                update.operation == "supersede"
                and (
                    base is None
                    or update.parent_revision_id != base.revision_id
                    or base.value == update.value
                )
            )
        ):
            raise LocalUpdatePolicyError("grounding_mismatch")
    return decision
