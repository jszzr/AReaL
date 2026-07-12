# SPDX-License-Identifier: Apache-2.0

"""Store-authentic V2 projection for provenance-aware Memory policies.

V1 deliberately exposes only user messages and feedback.  This opt-in V2 wire
adds the trajectory identity and ingest commitments needed to consume prior
tool interactions and verification outcomes without changing the V1 hash
domain or snapshot semantics.

The builder and public graph resolver are the local authenticity boundary: they
reload a durable SQLite evidence snapshot, recompute every event and snapshot
hash, and replay the complete base release.  The resulting DTO and graph remain
integrity commitments, not signatures.  Producer trust is a caller-supplied,
input-committed profile; a formal experiment must seal that input before future
outcomes exist.  The profile still assumes a trusted writer until an
authenticated ingress is implemented.

The two causal seals are external to this module and occur at different times:
the profile hash plus policy/projector version must be sealed before collecting
learning outcomes; after those historical outcomes are collected, the snapshot
ID/high-watermark and exact input hash must be sealed before any future task is
run.  ``observed_at`` is only a semantic-time filter.  It is not an availability
boundary and cannot replace the snapshot ingest high-watermark.  A formal
experiment also uses distinct identities and hash domains for the historical
claim verifier and the later task scorer.

Base replay accepts legacy plain-fact grounding only from the scope's explicit
application root.  Every later visible release must be the exact result of one
atomic Memory application.  That ledger binds the original snapshot, profile,
policy input, decision, revision, and result release; historical updates are
therefore replayed under their original profile rather than today's profile.
Each visible memory also commits an exclusive evidence-epoch boundary.  Only a
verified evaluator outcome newer than that memory's last update can justify its
next update.  Older claims, calls, and results may still be referenced by that
new outcome, which makes explicit re-verification possible without letting an
unrelated key consume or erase another key's evidence backlog.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from examples.memory_service import local_update_provenance as provenance

from areal.v2.memory_service import (
    EVIDENCE_SNAPSHOT_ORDERING_POLICY,
    EvidenceKind,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceSnapshotMember,
    EvidenceSnapshotSpec,
    MemoryApplicationReplayViewV1,
    MemoryApplicationUpdateProposal,
    MemoryApplicationV1,
    MemoryRelease,
    MemoryRevision,
    MemoryScope,
    RevisionOperation,
)
from areal.v2.memory_service.errors import MemoryServiceError
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

__all__ = [
    "BaseMemoryV2",
    "LocalUpdateProvenanceProjectionError",
    "POLICY_EVIDENCE_KINDS_V2",
    "PROVENANCE_PROJECTOR_ID_V1",
    "PROVENANCE_PROJECTOR_VERSION_SHA256_V1",
    "VERIFIED_CHAIN_POLICY_ID_V1",
    "VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1",
    "PolicyEvidenceV2",
    "PolicyInputV2",
    "ProvenanceGraphV1",
    "ProvenanceProfileV1",
    "VerifiedFactChainV1",
    "make_policy_input_v2",
    "policy_input_sha256_v2",
    "policy_input_wire_v2",
    "project_policy_input_from_snapshot_v2",
    "provenance_profile_sha256_v1",
    "provenance_profile_wire_v1",
    "provenance_application_context_v1",
    "canonical_verified_fact_grounding_v1",
    "resolve_provenance_graph_v1",
    "verified_chain_decision_sha256_v1",
]


class LocalUpdateProvenanceProjectionError(ValueError):
    """Stable, payload-free reason for rejecting projection state."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("projection error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ProvenanceProfileV1:
    schema_version: int
    fact_namespace: str
    agent_id: str
    agent_version_sha256: str
    tool_name: str
    tool_version_sha256: str
    evaluator_id: str
    evaluator_version_sha256: str


@dataclass(frozen=True, slots=True)
class PolicyEvidenceV2:
    evidence_id: str
    evidence_content_sha256: str
    ingest_order: int
    kind: str
    session_id: str
    run_id: str
    sequence_no: int
    observed_at_utc: str
    payload: str
    provenance_payload_sha256: str | None


@dataclass(frozen=True, slots=True)
class BaseMemoryV2:
    key: str
    value: str
    memory_id: str
    revision_id: str
    generation: int
    grounding_evidence: tuple[EvidenceSnapshotMember, ...]
    learning_evidence_after_ingest_order: int
    origin_application_id: str | None
    origin_application_content_sha256: str | None


@dataclass(frozen=True, slots=True)
class PolicyInputV2:
    """Constructible policy data; only a store-backed API authenticates it."""

    schema_version: int
    policy_scope_token: str
    base_release_id: str
    base_release_content_sha256: str
    provenance_profile: ProvenanceProfileV1
    provenance_profile_sha256: str
    learning_allowed_kinds: tuple[str, ...]
    evidence_snapshot_id: str
    evidence_snapshot_content_hash: str
    evidence_high_watermark: int
    evidence_ordering_policy: str
    evidence_snapshot_members: tuple[EvidenceSnapshotMember, ...]
    cutoff_utc: str
    evidence: tuple[PolicyEvidenceV2, ...]
    base_memories: tuple[BaseMemoryV2, ...]


@dataclass(frozen=True, slots=True)
class VerifiedFactChainV1:
    """A trusted-writer/profile match, not producer signature authentication."""

    fact_namespace: str
    key: str
    value: str
    call_id: str
    trajectory_id: str
    session_id: str
    run_id: str
    claim: EvidenceSnapshotMember
    tool_call: EvidenceSnapshotMember
    tool_result: EvidenceSnapshotMember
    outcome: EvidenceSnapshotMember


@dataclass(frozen=True, slots=True)
class ProvenanceGraphV1:
    verified_fact_chains: tuple[VerifiedFactChainV1, ...]
    rejected_outcome_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SnapshotMaterialV2:
    snapshot: EvidenceSnapshot
    evidence: tuple[PolicyEvidenceV2, ...]


@dataclass(frozen=True, slots=True)
class _ReleaseMaterialV1:
    release: MemoryRelease
    revisions: tuple[MemoryRevision, ...]


_PROFILE_SCHEMA_VERSION = 1
_INPUT_SCHEMA_VERSION = 2
_MAX_INT64 = 2**63 - 1
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_PATTERN = re.compile(r"evd_[0-9a-f]{24}")
_SNAPSHOT_ID_PATTERN = re.compile(r"esnap_[0-9a-f]{24}")
_RELEASE_ID_PATTERN = re.compile(r"rel_[0-9a-f]{24}")
_MEMORY_ID_PATTERN = re.compile(r"mem_[0-9a-f]{24}")
_REVISION_ID_PATTERN = re.compile(r"rev_[0-9a-f]{24}")
_APPLICATION_ID_PATTERN = re.compile(r"mapp_[0-9a-f]{24}")
_SCOPE_TOKEN_PATTERN = re.compile(r"scope_[0-9a-f]{64}")
_KEY_PATTERN = re.compile(r"project-[abcdefghjklmnpqrstuvwxyz23456789]{6}")
_VALUE_PATTERN = re.compile(r"[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5}")
_FACT_PATTERN = re.compile(
    r"(?P<key>project-[abcdefghjklmnpqrstuvwxyz23456789]{6}) = "
    r"(?P<value>[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5})"
)
_PROFILE_HASH_DOMAIN = b"areal-memory-provenance-profile-v1\0"
_INPUT_HASH_DOMAIN = b"areal-memory-provenance-policy-input-v2\0"
_SCOPE_TOKEN_DOMAIN = b"areal-memory-provenance-policy-scope-v2\0"
_APPLICATION_CONTEXT_KIND = "areal-memory-provenance-application-context-v1"

PROVENANCE_PROJECTOR_ID_V1 = "provenance-policy-input-v2"
PROVENANCE_PROJECTOR_VERSION_SHA256_V1 = hashlib.sha256(
    b"areal-memory-provenance-policy-input-projector-contract-v2-per-memory-outcome-epochs"
).hexdigest()
VERIFIED_CHAIN_POLICY_ID_V1 = "verified-chain-consensus-v1"
VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1 = hashlib.sha256(
    b"areal-memory-verified-chain-consensus-policy-contract-v2-per-memory-outcome-epochs"
).hexdigest()
_VERIFIED_CHAIN_DECISION_HASH_DOMAIN = (
    b"areal-memory-verified-chain-decision-v1\0"
)

POLICY_EVIDENCE_KINDS_V2 = tuple(
    sorted(
        (
            EvidenceKind.USER_MESSAGE.value,
            EvidenceKind.FEEDBACK.value,
            EvidenceKind.TOOL_CALL.value,
            EvidenceKind.TOOL_RESULT.value,
            EvidenceKind.OUTCOME.value,
        )
    )
)
_LEGACY_GROUNDING_KINDS = frozenset(
    (EvidenceKind.USER_MESSAGE.value, EvidenceKind.FEEDBACK.value)
)


def _error(reason: str) -> LocalUpdateProvenanceProjectionError:
    return LocalUpdateProvenanceProjectionError(reason)


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
        raise _error("closed_schema") from error


def _text(value: object, *, allow_blank: bool = False) -> str:
    if type(value) is not str:
        raise _error("closed_schema")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise _error("closed_schema") from error
    if not allow_blank and not value.strip():
        raise _error("closed_schema")
    return value


def _token(value: object) -> str:
    text = _text(value)
    if _TOKEN_PATTERN.fullmatch(text) is None:
        raise _error("closed_schema")
    return text


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise _error("closed_schema")
    return value


def _canonical_utc_text(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is None:
        raise _error("closed_schema")
    try:
        normalized = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise _error("closed_schema") from error
    return normalized.isoformat()


def _parse_canonical_utc_text(value: object) -> datetime:
    if type(value) is not str:
        raise _error("closed_schema")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise _error("closed_schema") from error
    if parsed.tzinfo is None or _canonical_utc_text(parsed) != value:
        raise _error("closed_schema")
    return parsed


def _validate_profile(value: object) -> ProvenanceProfileV1:
    if (
        type(value) is not ProvenanceProfileV1
        or type(value.schema_version) is not int
        or value.schema_version != _PROFILE_SCHEMA_VERSION
    ):
        raise _error("closed_schema")
    normalized = ProvenanceProfileV1(
        schema_version=_PROFILE_SCHEMA_VERSION,
        fact_namespace=_token(value.fact_namespace),
        agent_id=_token(value.agent_id),
        agent_version_sha256=_sha256(value.agent_version_sha256),
        tool_name=_token(value.tool_name),
        tool_version_sha256=_sha256(value.tool_version_sha256),
        evaluator_id=_token(value.evaluator_id),
        evaluator_version_sha256=_sha256(value.evaluator_version_sha256),
    )
    if normalized != value:
        raise _error("closed_schema")
    return value


def _profile_value(value: ProvenanceProfileV1) -> dict[str, object]:
    value = _validate_profile(value)
    return {
        "agent_id": value.agent_id,
        "agent_version_sha256": value.agent_version_sha256,
        "evaluator_id": value.evaluator_id,
        "evaluator_version_sha256": value.evaluator_version_sha256,
        "fact_namespace": value.fact_namespace,
        "schema_version": value.schema_version,
        "tool_name": value.tool_name,
        "tool_version_sha256": value.tool_version_sha256,
    }


def provenance_profile_wire_v1(value: ProvenanceProfileV1) -> bytes:
    return _canonical_json_bytes(_profile_value(value))


def provenance_profile_sha256_v1(value: ProvenanceProfileV1) -> str:
    return hashlib.sha256(
        _PROFILE_HASH_DOMAIN + provenance_profile_wire_v1(value)
    ).hexdigest()


def provenance_application_context_v1(value: ProvenanceProfileV1) -> str:
    """Commit the complete historical profile inside a core application."""

    profile = _validate_profile(value)
    return _canonical_json_bytes(
        {
            "kind": _APPLICATION_CONTEXT_KIND,
            "provenance_profile": _profile_value(profile),
            "provenance_profile_sha256": provenance_profile_sha256_v1(profile),
            "schema_version": 1,
        }
    ).decode("ascii")


def _profile_from_application_context(value: object) -> ProvenanceProfileV1:
    if type(value) is not str:
        raise _error("base_release_invalid")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise _error("base_release_invalid") from error
    if (
        type(decoded) is not dict
        or set(decoded)
        != {
            "kind",
            "provenance_profile",
            "provenance_profile_sha256",
            "schema_version",
        }
        or decoded["kind"] != _APPLICATION_CONTEXT_KIND
        or decoded["schema_version"] != 1
        or _canonical_json_bytes(decoded).decode("ascii") != value
        or type(decoded["provenance_profile"]) is not dict
    ):
        raise _error("base_release_invalid")
    profile_value = decoded["provenance_profile"]
    assert type(profile_value) is dict
    if set(profile_value) != {
        "agent_id",
        "agent_version_sha256",
        "evaluator_id",
        "evaluator_version_sha256",
        "fact_namespace",
        "schema_version",
        "tool_name",
        "tool_version_sha256",
    }:
        raise _error("base_release_invalid")
    try:
        profile = _validate_profile(ProvenanceProfileV1(**profile_value))
    except (TypeError, ValueError) as error:
        raise _error("base_release_invalid") from error
    if decoded["provenance_profile_sha256"] != provenance_profile_sha256_v1(profile):
        raise _error("base_release_invalid")
    return profile


def _scope_token(scope: MemoryScope) -> str:
    if type(scope) is not MemoryScope:
        raise _error("closed_schema")
    value = {
        "namespace": scope.namespace,
        "subject_id": scope.subject_id,
        "tenant_id": scope.tenant_id,
    }
    return (
        "scope_"
        + hashlib.sha256(_SCOPE_TOKEN_DOMAIN + _canonical_json_bytes(value)).hexdigest()
    )


def _validate_member(value: object) -> EvidenceSnapshotMember:
    if (
        type(value) is not EvidenceSnapshotMember
        or type(value.evidence_id) is not str
        or _EVIDENCE_ID_PATTERN.fullmatch(value.evidence_id) is None
        or type(value.evidence_content_hash) is not str
        or _SHA256_PATTERN.fullmatch(value.evidence_content_hash) is None
        or type(value.ingest_order) is not int
        or not 0 <= value.ingest_order <= _MAX_INT64
    ):
        raise _error("closed_schema")
    if value.evidence_id != f"evd_{value.evidence_content_hash[:24]}":
        raise _error("input_invariant")
    return value


def _parsed_provenance_sha256(
    *,
    kind: str,
    payload: str,
) -> str | None:
    try:
        parsed = provenance.parse_evidence_provenance_payload_v1(kind, payload)
    except provenance.LocalUpdateProvenanceError:
        return None
    return provenance.provenance_payload_sha256_v1(parsed)


def _validate_evidence(value: object) -> PolicyEvidenceV2:
    if (
        type(value) is not PolicyEvidenceV2
        or type(value.evidence_id) is not str
        or _EVIDENCE_ID_PATTERN.fullmatch(value.evidence_id) is None
        or type(value.evidence_content_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.evidence_content_sha256) is None
        or type(value.ingest_order) is not int
        or not 0 <= value.ingest_order <= _MAX_INT64
        or type(value.kind) is not str
        or value.kind not in POLICY_EVIDENCE_KINDS_V2
        or type(value.sequence_no) is not int
        or not 0 <= value.sequence_no <= _MAX_INT64
        or (
            value.provenance_payload_sha256 is not None
            and (
                type(value.provenance_payload_sha256) is not str
                or _SHA256_PATTERN.fullmatch(value.provenance_payload_sha256) is None
            )
        )
    ):
        raise _error("closed_schema")
    if value.evidence_id != f"evd_{value.evidence_content_sha256[:24]}":
        raise _error("input_invariant")
    _text(value.session_id)
    _text(value.run_id)
    _parse_canonical_utc_text(value.observed_at_utc)
    payload = _text(value.payload, allow_blank=True)
    expected_provenance_hash = _parsed_provenance_sha256(
        kind=value.kind,
        payload=payload,
    )
    if value.provenance_payload_sha256 != expected_provenance_hash:
        raise _error("input_invariant")
    return value


def _validate_base_memory(value: object) -> BaseMemoryV2:
    if (
        type(value) is not BaseMemoryV2
        or type(value.key) is not str
        or _KEY_PATTERN.fullmatch(value.key) is None
        or type(value.value) is not str
        or _VALUE_PATTERN.fullmatch(value.value) is None
        or type(value.memory_id) is not str
        or _MEMORY_ID_PATTERN.fullmatch(value.memory_id) is None
        or type(value.revision_id) is not str
        or _REVISION_ID_PATTERN.fullmatch(value.revision_id) is None
        or type(value.generation) is not int
        or not 0 <= value.generation <= _MAX_INT64
        or type(value.grounding_evidence) is not tuple
        or not value.grounding_evidence
        or type(value.learning_evidence_after_ingest_order) is not int
        or not -1
        <= value.learning_evidence_after_ingest_order
        <= _MAX_INT64
        or (
            value.origin_application_id is None
            and (
                value.origin_application_content_sha256 is not None
                or value.learning_evidence_after_ingest_order != -1
            )
        )
        or (
            value.origin_application_id is not None
            and (
                type(value.origin_application_id) is not str
                or _APPLICATION_ID_PATTERN.fullmatch(value.origin_application_id)
                is None
                or type(value.origin_application_content_sha256) is not str
                or _SHA256_PATTERN.fullmatch(value.origin_application_content_sha256)
                is None
                or value.origin_application_id
                != f"mapp_{value.origin_application_content_sha256[:24]}"
            )
        )
    ):
        raise _error("closed_schema")
    grounding = tuple(_validate_member(item) for item in value.grounding_evidence)
    if (
        len({item.evidence_id for item in grounding}) != len(grounding)
        or (
            value.origin_application_id is not None
            and any(
                item.ingest_order > value.learning_evidence_after_ingest_order
                for item in grounding
            )
        )
    ):
        raise _error("input_invariant")
    return value


def _chain_members(
    value: VerifiedFactChainV1,
) -> tuple[EvidenceSnapshotMember, ...]:
    return (value.claim, value.tool_call, value.tool_result, value.outcome)


def _stable_unique_chain_members(
    chains: tuple[VerifiedFactChainV1, ...],
) -> tuple[EvidenceSnapshotMember, ...]:
    """Flatten chains without repeating a shared claim or other member."""

    result: list[EvidenceSnapshotMember] = []
    seen: set[str] = set()
    for chain in chains:
        for member in _chain_members(chain):
            if member.evidence_id not in seen:
                seen.add(member.evidence_id)
                result.append(member)
    return tuple(result)


def _canonical_grounding_for_fact(
    *,
    profile: ProvenanceProfileV1,
    key: str,
    fact_value: str,
    graph: ProvenanceGraphV1,
) -> tuple[EvidenceSnapshotMember, ...]:
    key_chains = tuple(
        chain
        for chain in graph.verified_fact_chains
        if chain.fact_namespace == profile.fact_namespace and chain.key == key
    )
    if not key_chains or {chain.value for chain in key_chains} != {fact_value}:
        return ()
    return _stable_unique_chain_members(key_chains)


def canonical_verified_fact_grounding_v1(
    *,
    profile: ProvenanceProfileV1,
    graph: ProvenanceGraphV1,
    key: str,
    fact_value: str,
) -> tuple[EvidenceSnapshotMember, ...]:
    """Return the one consensus grounding or an empty abstention."""

    profile = _validate_profile(profile)
    if (
        type(graph) is not ProvenanceGraphV1
        or type(key) is not str
        or _KEY_PATTERN.fullmatch(key) is None
        or type(fact_value) is not str
        or _VALUE_PATTERN.fullmatch(fact_value) is None
    ):
        raise _error("closed_schema")
    return _canonical_grounding_for_fact(
        profile=profile,
        key=key,
        fact_value=fact_value,
        graph=graph,
    )


def _validate_policy_input(value: object) -> PolicyInputV2:
    if (
        type(value) is not PolicyInputV2
        or type(value.schema_version) is not int
        or value.schema_version != _INPUT_SCHEMA_VERSION
        or type(value.policy_scope_token) is not str
        or _SCOPE_TOKEN_PATTERN.fullmatch(value.policy_scope_token) is None
        or type(value.base_release_id) is not str
        or _RELEASE_ID_PATTERN.fullmatch(value.base_release_id) is None
        or type(value.base_release_content_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.base_release_content_sha256) is None
        or type(value.provenance_profile_sha256) is not str
        or _SHA256_PATTERN.fullmatch(value.provenance_profile_sha256) is None
        or type(value.learning_allowed_kinds) is not tuple
        or any(type(kind) is not str for kind in value.learning_allowed_kinds)
        or value.learning_allowed_kinds != POLICY_EVIDENCE_KINDS_V2
        or type(value.evidence_snapshot_id) is not str
        or _SNAPSHOT_ID_PATTERN.fullmatch(value.evidence_snapshot_id) is None
        or type(value.evidence_snapshot_content_hash) is not str
        or _SHA256_PATTERN.fullmatch(value.evidence_snapshot_content_hash) is None
        or type(value.evidence_high_watermark) is not int
        or not -1 <= value.evidence_high_watermark <= _MAX_INT64
        or type(value.evidence_ordering_policy) is not str
        or value.evidence_ordering_policy != EVIDENCE_SNAPSHOT_ORDERING_POLICY
        or type(value.evidence_snapshot_members) is not tuple
        or type(value.evidence) is not tuple
        or type(value.base_memories) is not tuple
    ):
        raise _error("closed_schema")
    profile = _validate_profile(value.provenance_profile)
    if value.provenance_profile_sha256 != provenance_profile_sha256_v1(profile):
        raise _error("input_invariant")
    cutoff = _parse_canonical_utc_text(value.cutoff_utc)
    members = tuple(_validate_member(item) for item in value.evidence_snapshot_members)
    evidence = tuple(_validate_evidence(item) for item in value.evidence)
    bases = tuple(_validate_base_memory(item) for item in value.base_memories)
    member_by_id = {item.evidence_id: item for item in members}
    if (
        value.base_release_id != f"rel_{value.base_release_content_sha256[:24]}"
        or value.evidence_snapshot_id
        != f"esnap_{value.evidence_snapshot_content_hash[:24]}"
        or len(member_by_id) != len(members)
        or len({item.ingest_order for item in members}) != len(members)
        or any(item.ingest_order > value.evidence_high_watermark for item in members)
        or tuple(item.evidence_id for item in members)
        != tuple(item.evidence_id for item in evidence)
        or any(
            member.evidence_content_hash != item.evidence_content_sha256
            or member.ingest_order != item.ingest_order
            for member, item in zip(members, evidence, strict=True)
        )
        or any(
            _parse_canonical_utc_text(item.observed_at_utc) > cutoff
            for item in evidence
        )
        or bases != tuple(sorted(bases, key=lambda item: item.key))
        or len({item.key for item in bases}) != len(bases)
        or len({item.memory_id for item in bases}) != len(bases)
        or len({item.revision_id for item in bases}) != len(bases)
        or any(
            member_by_id.get(ground.evidence_id) != ground
            for base in bases
            for ground in base.grounding_evidence
        )
        or any(
            base.learning_evidence_after_ingest_order
            > value.evidence_high_watermark
            for base in bases
        )
    ):
        raise _error("input_invariant")
    return value


def _member_value(value: EvidenceSnapshotMember) -> dict[str, object]:
    value = _validate_member(value)
    return {
        "evidence_content_hash": value.evidence_content_hash,
        "evidence_id": value.evidence_id,
        "ingest_order": value.ingest_order,
    }


def _input_value(value: PolicyInputV2) -> dict[str, object]:
    value = _validate_policy_input(value)
    return {
        "base_memories": [
            {
                "generation": item.generation,
                "grounding_evidence": [
                    _member_value(member) for member in item.grounding_evidence
                ],
                "key": item.key,
                "learning_evidence_after_ingest_order": (
                    item.learning_evidence_after_ingest_order
                ),
                "memory_id": item.memory_id,
                "origin_application_content_sha256": (
                    item.origin_application_content_sha256
                ),
                "origin_application_id": item.origin_application_id,
                "revision_id": item.revision_id,
                "value": item.value,
            }
            for item in value.base_memories
        ],
        "base_release_content_sha256": value.base_release_content_sha256,
        "base_release_id": value.base_release_id,
        "cutoff_utc": value.cutoff_utc,
        "evidence": [
            {
                "evidence_content_sha256": item.evidence_content_sha256,
                "evidence_id": item.evidence_id,
                "ingest_order": item.ingest_order,
                "kind": item.kind,
                "observed_at_utc": item.observed_at_utc,
                "payload": item.payload,
                "provenance_payload_sha256": item.provenance_payload_sha256,
                "run_id": item.run_id,
                "sequence_no": item.sequence_no,
                "session_id": item.session_id,
            }
            for item in value.evidence
        ],
        "evidence_high_watermark": value.evidence_high_watermark,
        "evidence_ordering_policy": value.evidence_ordering_policy,
        "evidence_snapshot_content_hash": value.evidence_snapshot_content_hash,
        "evidence_snapshot_id": value.evidence_snapshot_id,
        "evidence_snapshot_members": [
            _member_value(member) for member in value.evidence_snapshot_members
        ],
        "learning_allowed_kinds": list(value.learning_allowed_kinds),
        "policy_scope_token": value.policy_scope_token,
        "provenance_profile": _profile_value(value.provenance_profile),
        "provenance_profile_sha256": value.provenance_profile_sha256,
        "schema_version": value.schema_version,
    }


def policy_input_wire_v2(value: PolicyInputV2) -> bytes:
    return _canonical_json_bytes(_input_value(value))


def policy_input_sha256_v2(value: PolicyInputV2) -> str:
    return hashlib.sha256(_INPUT_HASH_DOMAIN + policy_input_wire_v2(value)).hexdigest()


def verified_chain_decision_sha256_v1(
    source_input: PolicyInputV2,
    updates: tuple[MemoryApplicationUpdateProposal, ...],
) -> str:
    """Hash the exact input and ordered changed updates of the V1 policy."""

    input_sha256 = policy_input_sha256_v2(source_input)
    if (
        type(updates) is not tuple
        or not updates
        or any(type(item) is not MemoryApplicationUpdateProposal for item in updates)
    ):
        raise _error("closed_schema")
    value = {
        "input_sha256": input_sha256,
        "policy_id": VERIFIED_CHAIN_POLICY_ID_V1,
        "policy_version_sha256": VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1,
        "schema_version": 1,
        "updates": [
            {
                "content": item.content,
                "evidence_ids": list(item.evidence_ids),
                "operation": item.operation.value,
                "parent_revision_id": item.parent_revision_id,
            }
            for item in updates
        ],
    }
    return hashlib.sha256(
        _VERIFIED_CHAIN_DECISION_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _load_snapshot_projection(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    snapshot_id: str,
) -> tuple[EvidenceSnapshot, tuple[EvidenceRecord, ...]]:
    try:
        snapshot = store.get_evidence_snapshot(scope, snapshot_id)
        records = store.get_evidence_snapshot_evidence(scope, snapshot_id)
        if type(snapshot) is not EvidenceSnapshot:
            raise _error("evidence_snapshot_invalid")
        expected_spec = EvidenceSnapshotSpec(
            scope=scope,
            allowed_kinds=tuple(
                EvidenceKind(kind) for kind in POLICY_EVIDENCE_KINDS_V2
            ),
            cutoff=snapshot.spec.cutoff,
        )
        expected_hash = hashlib.sha256(snapshot.canonical_bytes()).hexdigest()
        if (
            snapshot.snapshot_id != snapshot_id
            or snapshot.spec != expected_spec
            or snapshot.content_hash != expected_hash
            or snapshot.snapshot_id != f"esnap_{expected_hash[:24]}"
            or type(records) is not tuple
            or tuple((record.evidence_id, record.content_hash) for record in records)
            != tuple(
                (member.evidence_id, member.evidence_content_hash)
                for member in snapshot.members
            )
            or any(
                record.event.scope != scope
                or record.event.kind.value not in POLICY_EVIDENCE_KINDS_V2
                or record.event.observed_at > snapshot.spec.cutoff
                or record.content_hash
                != hashlib.sha256(record.event.canonical_bytes()).hexdigest()
                or record.evidence_id != f"evd_{record.content_hash[:24]}"
                for record in records
            )
        ):
            raise _error("evidence_snapshot_invalid")
    except LocalUpdateProvenanceProjectionError:
        raise
    except (
        AttributeError,
        MemoryServiceError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise _error("evidence_snapshot_invalid") from error
    return snapshot, records


def _snapshot_material_from_values_v2(
    *,
    scope: MemoryScope,
    snapshot: EvidenceSnapshot,
    records: tuple[EvidenceRecord, ...],
) -> _SnapshotMaterialV2:
    expected_spec = EvidenceSnapshotSpec(
        scope=scope,
        allowed_kinds=tuple(EvidenceKind(kind) for kind in POLICY_EVIDENCE_KINDS_V2),
        cutoff=snapshot.spec.cutoff,
    )
    expected_hash = hashlib.sha256(snapshot.canonical_bytes()).hexdigest()
    if (
        type(snapshot) is not EvidenceSnapshot
        or type(records) is not tuple
        or any(type(record) is not EvidenceRecord for record in records)
        or snapshot.spec != expected_spec
        or snapshot.content_hash != expected_hash
        or snapshot.snapshot_id != f"esnap_{expected_hash[:24]}"
        or tuple((record.evidence_id, record.content_hash) for record in records)
        != tuple(
            (member.evidence_id, member.evidence_content_hash)
            for member in snapshot.members
        )
        or any(
            record.event.scope != scope
            or record.event.kind.value not in POLICY_EVIDENCE_KINDS_V2
            or record.event.observed_at > snapshot.spec.cutoff
            or record.content_hash
            != hashlib.sha256(record.event.canonical_bytes()).hexdigest()
            or record.evidence_id != f"evd_{record.content_hash[:24]}"
            for record in records
        )
    ):
        raise _error("evidence_snapshot_invalid")
    evidence = tuple(
        PolicyEvidenceV2(
            evidence_id=record.evidence_id,
            evidence_content_sha256=record.content_hash,
            ingest_order=member.ingest_order,
            kind=record.event.kind.value,
            session_id=record.event.session_id,
            run_id=record.event.run_id,
            sequence_no=record.event.sequence_no,
            observed_at_utc=_canonical_utc_text(record.event.observed_at),
            payload=record.event.payload,
            provenance_payload_sha256=_parsed_provenance_sha256(
                kind=record.event.kind.value,
                payload=record.event.payload,
            ),
        )
        for member, record in zip(snapshot.members, records, strict=True)
    )
    return _SnapshotMaterialV2(snapshot=snapshot, evidence=evidence)


def _load_snapshot_material_v2(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    snapshot_id: str,
) -> _SnapshotMaterialV2:
    snapshot, records = _load_snapshot_projection(
        store=store,
        scope=scope,
        snapshot_id=snapshot_id,
    )
    return _snapshot_material_from_values_v2(
        scope=scope,
        snapshot=snapshot,
        records=records,
    )


def _load_legacy_revision_fact_v2(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    revision: MemoryRevision,
    member_by_id: dict[str, EvidenceSnapshotMember],
    evidence_by_id: dict[str, PolicyEvidenceV2],
) -> tuple[str, str, tuple[EvidenceSnapshotMember, ...]]:
    candidate = store.get_candidate(scope, revision.proposal.candidate_id)
    evidence_records = store.get_candidate_evidence(scope, candidate.candidate_id)
    match = _FACT_PATTERN.fullmatch(candidate.proposal.content)
    if match is None:
        raise _error("base_release_invalid")
    grounding = tuple(member_by_id[record.evidence_id] for record in evidence_records)
    key = match.group("key")
    fact_value = match.group("value")
    if (
        revision.proposal.scope != scope
        or candidate.proposal.scope != scope
        or revision.proposal.candidate_id != candidate.candidate_id
        or candidate.proposal.evidence_ids
        != tuple(record.evidence_id for record in evidence_records)
        or not evidence_records
        or any(record.event.scope != scope for record in evidence_records)
        or any(
            member.evidence_content_hash != record.content_hash
            for member, record in zip(grounding, evidence_records, strict=True)
        )
        or any(
            evidence_by_id[member.evidence_id].kind not in _LEGACY_GROUNDING_KINDS
            or evidence_by_id[member.evidence_id].payload != f"{key} = {fact_value}"
            for member in grounding
        )
    ):
        raise _error("base_release_invalid")
    return key, fact_value, grounding


def _project_root_memories(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    release: MemoryRelease,
    revisions: tuple[MemoryRevision, ...],
    members: tuple[EvidenceSnapshotMember, ...],
    evidence: tuple[PolicyEvidenceV2, ...],
) -> tuple[BaseMemoryV2, ...]:
    member_by_id = {member.evidence_id: member for member in members}
    evidence_by_id = {item.evidence_id: item for item in evidence}
    bases: list[BaseMemoryV2] = []
    for revision in revisions:
        current = revision
        expected_generation = revision.generation
        lineage_revision_ids: set[str] = set()
        exposed_fact: tuple[str, str] | None = None
        exposed_grounding: tuple[EvidenceSnapshotMember, ...] | None = None
        while True:
            if (
                type(current) is not MemoryRevision
                or current.revision_id in lineage_revision_ids
                or current.memory_id != revision.memory_id
                or current.generation != expected_generation
            ):
                raise _error("base_release_invalid")
            lineage_revision_ids.add(current.revision_id)
            key, fact_value, grounding = _load_legacy_revision_fact_v2(
                store=store,
                scope=scope,
                revision=current,
                member_by_id=member_by_id,
                evidence_by_id=evidence_by_id,
            )
            if exposed_fact is None:
                exposed_fact = (key, fact_value)
                exposed_grounding = grounding
            if current.proposal.operation is RevisionOperation.ADD:
                if current.proposal.parent_revision_id is not None:
                    raise _error("base_release_invalid")
                break
            if (
                current.proposal.operation is not RevisionOperation.SUPERSEDE
                or current.proposal.parent_revision_id is None
                or expected_generation <= 0
            ):
                raise _error("base_release_invalid")
            parent = store.get_revision(scope, current.proposal.parent_revision_id)
            if (
                parent.revision_id != current.proposal.parent_revision_id
                or parent.memory_id != current.memory_id
                or parent.generation != expected_generation - 1
            ):
                raise _error("base_release_invalid")
            current = parent
            expected_generation -= 1
        if (
            expected_generation != 0
            or exposed_fact is None
            or exposed_grounding is None
        ):
            raise _error("base_release_invalid")
        key, fact_value = exposed_fact
        bases.append(
            BaseMemoryV2(
                key=key,
                value=fact_value,
                memory_id=revision.memory_id,
                revision_id=revision.revision_id,
                generation=revision.generation,
                grounding_evidence=exposed_grounding,
                learning_evidence_after_ingest_order=-1,
                origin_application_id=None,
                origin_application_content_sha256=None,
            )
        )
    if (
        release.manifest.scope != scope
        or tuple(revision.revision_id for revision in revisions)
        != release.manifest.revision_ids
    ):
        raise _error("base_release_invalid")
    return tuple(sorted(bases, key=lambda value: value.key))


def _authenticate_application_updates(
    *,
    application: MemoryApplicationV1,
    source_input: PolicyInputV2,
) -> dict[str, BaseMemoryV2]:
    profile = source_input.provenance_profile
    graph = _resolve_provenance_graph_from_input_v1(source_input)
    source_by_revision = {base.revision_id: base for base in source_input.base_memories}
    source_by_key = {base.key: base for base in source_input.base_memories}
    updates: dict[str, BaseMemoryV2] = {}
    for proposal_update, applied_update in zip(
        application.proposal.updates,
        application.applied_updates,
        strict=True,
    ):
        match = _FACT_PATTERN.fullmatch(proposal_update.content)
        if match is None:
            raise _error("base_release_invalid")
        key = match.group("key")
        fact_value = match.group("value")
        expected_grounding = _canonical_grounding_for_fact(
            profile=profile,
            key=key,
            fact_value=fact_value,
            graph=graph,
        )
        if (
            not expected_grounding
            or proposal_update.evidence_ids
            != tuple(member.evidence_id for member in expected_grounding)
            or applied_update.grounding != expected_grounding
            or applied_update.operation is not proposal_update.operation
            or applied_update.parent_revision_id != proposal_update.parent_revision_id
        ):
            raise _error("base_release_invalid")
        if proposal_update.operation is RevisionOperation.SUPERSEDE:
            assert proposal_update.parent_revision_id is not None
            parent = source_by_revision.get(proposal_update.parent_revision_id)
            if parent is None or parent.key != key or parent.value == fact_value:
                raise _error("base_release_invalid")
        elif proposal_update.operation is RevisionOperation.ADD:
            if key in source_by_key:
                raise _error("base_release_invalid")
        else:
            raise _error("base_release_invalid")
        updates[applied_update.revision_id] = BaseMemoryV2(
            key=key,
            value=fact_value,
            memory_id=applied_update.memory_id,
            revision_id=applied_update.revision_id,
            generation=applied_update.generation,
            grounding_evidence=applied_update.grounding,
            learning_evidence_after_ingest_order=(
                application.source_evidence_high_watermark
            ),
            origin_application_id=application.application_id,
            origin_application_content_sha256=application.content_hash,
        )
    if len(updates) != len(application.applied_updates):
        raise _error("base_release_invalid")
    return updates


def _build_policy_input_v2(
    *,
    scope: MemoryScope,
    release: MemoryRelease,
    bases: tuple[BaseMemoryV2, ...],
    material: _SnapshotMaterialV2,
    profile: ProvenanceProfileV1,
) -> PolicyInputV2:
    snapshot = material.snapshot
    return _validate_policy_input(
        PolicyInputV2(
            schema_version=_INPUT_SCHEMA_VERSION,
            policy_scope_token=_scope_token(scope),
            base_release_id=release.release_id,
            base_release_content_sha256=release.content_hash,
            provenance_profile=profile,
            provenance_profile_sha256=provenance_profile_sha256_v1(profile),
            learning_allowed_kinds=POLICY_EVIDENCE_KINDS_V2,
            evidence_snapshot_id=snapshot.snapshot_id,
            evidence_snapshot_content_hash=snapshot.content_hash,
            evidence_high_watermark=snapshot.evidence_high_watermark,
            evidence_ordering_policy=snapshot.ordering_policy,
            evidence_snapshot_members=snapshot.members,
            cutoff_utc=_canonical_utc_text(snapshot.spec.cutoff),
            evidence=material.evidence,
            base_memories=bases,
        )
    )


def _apply_application_result_v1(
    *,
    application: MemoryApplicationV1,
    result: _ReleaseMaterialV1,
    source_input: PolicyInputV2,
) -> tuple[BaseMemoryV2, ...]:
    if (
        application.result_release_id != result.release.release_id
        or application.result_release_content_sha256 != result.release.content_hash
        or application.result_revision_ids
        != result.release.manifest.revision_ids
    ):
        raise _error("base_release_invalid")
    updated = _authenticate_application_updates(
        application=application,
        source_input=source_input,
    )
    source_by_revision = {
        base.revision_id: base for base in source_input.base_memories
    }
    ordered_bases: list[BaseMemoryV2] = []
    for revision in result.revisions:
        base = updated.get(revision.revision_id)
        if base is None:
            base = source_by_revision.get(revision.revision_id)
        if base is None:
            raise _error("base_release_invalid")
        ordered_bases.append(base)
    if len({base.revision_id for base in ordered_bases}) != len(ordered_bases):
        raise _error("base_release_invalid")
    return tuple(sorted(ordered_bases, key=lambda value: value.key))


def _project_base_memories(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    base_release_id: str,
    members: tuple[EvidenceSnapshotMember, ...],
    evidence: tuple[PolicyEvidenceV2, ...],
) -> tuple[MemoryRelease, tuple[BaseMemoryV2, ...]]:
    """Replay one bulk-loaded linear ledger from its root to the target."""

    try:
        replay = store.get_memory_application_replay(scope, base_release_id)
        if (
            type(replay) is not MemoryApplicationReplayViewV1
            or replay.root.scope != scope
            or replay.target_release.release_id != base_release_id
        ):
            raise _error("base_release_invalid")
        current = _ReleaseMaterialV1(
            release=replay.root_release,
            revisions=replay.root_revisions,
        )
        if not replay.steps:
            bases = _project_root_memories(
                store=store,
                scope=scope,
                release=current.release,
                revisions=current.revisions,
                members=members,
                evidence=evidence,
            )
            return current.release, bases

        current_bases: tuple[BaseMemoryV2, ...] | None = None
        material_by_snapshot_id: dict[str, _SnapshotMaterialV2] = {}
        for step in replay.steps:
            application = step.application
            if (
                application.proposal.projector_id != PROVENANCE_PROJECTOR_ID_V1
                or application.proposal.projector_version_sha256
                != PROVENANCE_PROJECTOR_VERSION_SHA256_V1
            ):
                raise _error("base_release_invalid")
            historical_profile = _profile_from_application_context(
                application.proposal.policy_context
            )
            material = material_by_snapshot_id.get(step.source_snapshot.snapshot_id)
            if material is None:
                material = _snapshot_material_from_values_v2(
                    scope=scope,
                    snapshot=step.source_snapshot,
                    records=step.source_evidence,
                )
                material_by_snapshot_id[step.source_snapshot.snapshot_id] = material
            if current_bases is None:
                current_bases = _project_root_memories(
                    store=store,
                    scope=scope,
                    release=current.release,
                    revisions=current.revisions,
                    members=material.snapshot.members,
                    evidence=material.evidence,
                )
            if (
                application.proposal.source_base_release_id
                != current.release.release_id
                or application.base_release_content_sha256
                != current.release.content_hash
                or application.source_snapshot_content_sha256
                != material.snapshot.content_hash
                or application.source_evidence_high_watermark
                != material.snapshot.evidence_high_watermark
            ):
                raise _error("base_release_invalid")
            source_input = _build_policy_input_v2(
                scope=scope,
                release=current.release,
                bases=current_bases,
                material=material,
                profile=historical_profile,
            )
            if policy_input_sha256_v2(source_input) != (
                application.proposal.policy_input_sha256
            ):
                raise _error("base_release_invalid")
            if (
                application.proposal.policy_id != VERIFIED_CHAIN_POLICY_ID_V1
                or application.proposal.policy_version_sha256
                != VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1
                or application.proposal.decision_sha256
                != verified_chain_decision_sha256_v1(
                    source_input,
                    application.proposal.updates,
                )
            ):
                raise _error("base_release_invalid")
            result = _ReleaseMaterialV1(
                release=step.result_release,
                revisions=step.result_revisions,
            )
            current_bases = _apply_application_result_v1(
                application=application,
                result=result,
                source_input=source_input,
            )
            current = result
        if (
            current_bases is None
            or current.release.release_id != replay.target_release.release_id
            or current.release.content_hash != replay.target_release.content_hash
        ):
            raise _error("base_release_invalid")
    except LocalUpdateProvenanceProjectionError:
        raise
    except (
        AttributeError,
        KeyError,
        MemoryServiceError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise _error("base_release_invalid") from error
    return current.release, current_bases


def _project_policy_input_from_snapshot_v2(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    base_release_id: str,
    evidence_snapshot_id: str,
    provenance_profile: ProvenanceProfileV1,
) -> PolicyInputV2:
    """Reconstruct a V2 policy input from one store-authentic snapshot."""

    if (
        type(store) is not SQLiteMemoryStore
        or type(scope) is not MemoryScope
        or type(base_release_id) is not str
        or _RELEASE_ID_PATTERN.fullmatch(base_release_id) is None
        or type(evidence_snapshot_id) is not str
        or _SNAPSHOT_ID_PATTERN.fullmatch(evidence_snapshot_id) is None
    ):
        raise _error("closed_schema")
    profile = _validate_profile(provenance_profile)
    material = _load_snapshot_material_v2(
        store=store,
        scope=scope,
        snapshot_id=evidence_snapshot_id,
    )
    release, bases = _project_base_memories(
        store=store,
        scope=scope,
        base_release_id=base_release_id,
        members=material.snapshot.members,
        evidence=material.evidence,
    )
    return _build_policy_input_v2(
        scope=scope,
        release=release,
        bases=bases,
        material=material,
        profile=profile,
    )


def project_policy_input_from_snapshot_v2(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    base_release_id: str,
    evidence_snapshot_id: str,
    provenance_profile: ProvenanceProfileV1,
) -> PolicyInputV2:
    """Reconstruct one input and replay its committed application lineage."""

    return _project_policy_input_from_snapshot_v2(
        store=store,
        scope=scope,
        base_release_id=base_release_id,
        evidence_snapshot_id=evidence_snapshot_id,
        provenance_profile=provenance_profile,
    )


def make_policy_input_v2(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    base_release_id: str,
    cutoff: datetime,
    evidence_snapshot_idempotency_key: str,
    provenance_profile: ProvenanceProfileV1,
) -> PolicyInputV2:
    """Validate the base, seal complete V2 evidence, then project it.

    Calling this function at the wrong time does not create a causal experiment:
    a caller could invoke it after a future outcome already exists.  The runner
    must pre-seal the profile before learning outcomes and must seal this
    function's snapshot high-watermark/input hash before future tasks.
    """

    if (
        type(store) is not SQLiteMemoryStore
        or type(scope) is not MemoryScope
        or type(base_release_id) is not str
        or _RELEASE_ID_PATTERN.fullmatch(base_release_id) is None
        or type(evidence_snapshot_idempotency_key) is not str
        or not evidence_snapshot_idempotency_key.strip()
    ):
        raise _error("closed_schema")
    try:
        evidence_snapshot_idempotency_key.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise _error("closed_schema") from error
    profile = _validate_profile(provenance_profile)
    cutoff_text = _canonical_utc_text(cutoff)
    cutoff_utc = _parse_canonical_utc_text(cutoff_text)

    # Reject an absent or structurally mismatched release before the durable
    # snapshot write.  Full lineage grounding requires that snapshot's exact
    # members and is replayed immediately afterward by the projector.
    try:
        base_release = store.get_release(scope, base_release_id)
        base_revisions = store.get_release_revisions(scope, base_release_id)
        if (
            base_release.release_id != base_release_id
            or base_release.manifest.scope != scope
            or base_release.manifest.revision_ids
            != tuple(revision.revision_id for revision in base_revisions)
        ):
            raise _error("base_release_invalid")
    except LocalUpdateProvenanceProjectionError:
        raise
    except (AttributeError, MemoryServiceError, TypeError, ValueError) as error:
        raise _error("base_release_invalid") from error
    try:
        snapshot = store.seal_evidence_snapshot(
            EvidenceSnapshotSpec(
                scope=scope,
                allowed_kinds=tuple(
                    EvidenceKind(kind) for kind in POLICY_EVIDENCE_KINDS_V2
                ),
                cutoff=cutoff_utc,
            ),
            idempotency_key=evidence_snapshot_idempotency_key,
        )
        if type(snapshot) is not EvidenceSnapshot:
            raise _error("evidence_snapshot_invalid")
    except LocalUpdateProvenanceProjectionError:
        raise
    except (
        AttributeError,
        MemoryServiceError,
        OverflowError,
        TypeError,
        ValueError,
    ) as error:
        raise _error("evidence_snapshot_invalid") from error
    result = project_policy_input_from_snapshot_v2(
        store=store,
        scope=scope,
        base_release_id=base_release_id,
        evidence_snapshot_id=snapshot.snapshot_id,
        provenance_profile=profile,
    )
    if result.cutoff_utc != cutoff_text:
        raise _error("evidence_snapshot_invalid")
    return result


def _parse_evidence_payload(
    value: PolicyEvidenceV2,
) -> provenance.ProvenancePayloadV1 | None:
    if value.provenance_payload_sha256 is None:
        return None
    try:
        parsed = provenance.parse_evidence_provenance_payload_v1(
            value.kind,
            value.payload,
        )
    except provenance.LocalUpdateProvenanceError as error:
        raise _error("input_invariant") from error
    if (
        provenance.provenance_payload_sha256_v1(parsed)
        != value.provenance_payload_sha256
    ):
        raise _error("input_invariant")
    # V2 defines one trajectory as one run.  A future cross-run trajectory
    # profile needs a new wire instead of silently weakening this join.
    if parsed.trajectory_id != value.run_id:
        return None
    return parsed


def _profile_matches_outcome(
    profile: ProvenanceProfileV1,
    payload: provenance.ProvenancePayloadV1,
) -> bool:
    body = payload.body
    producer = payload.producer
    return (
        type(body) is provenance.VerificationOutcomeBodyV1
        and producer.kind is provenance.ProducerKindV1.EVALUATOR
        and producer.producer_id == profile.evaluator_id
        and producer.version_sha256 == profile.evaluator_version_sha256
        and body.evaluator_id == profile.evaluator_id
        and body.evaluator_version_sha256 == profile.evaluator_version_sha256
    )


def _profile_matches_result(
    profile: ProvenanceProfileV1,
    payload: provenance.ProvenancePayloadV1,
) -> bool:
    body = payload.body
    producer = payload.producer
    return (
        type(body) is provenance.ToolResultBodyV1
        and producer.kind is provenance.ProducerKindV1.TOOL
        and producer.producer_id == profile.tool_name
        and producer.version_sha256 == profile.tool_version_sha256
        and body.tool_name == profile.tool_name
        and body.tool_version_sha256 == profile.tool_version_sha256
        and body.fact_namespace == profile.fact_namespace
    )


def _profile_matches_call(
    profile: ProvenanceProfileV1,
    payload: provenance.ProvenancePayloadV1,
) -> bool:
    body = payload.body
    producer = payload.producer
    return (
        type(body) is provenance.ToolCallBodyV1
        and producer.kind is provenance.ProducerKindV1.AGENT
        and producer.producer_id == profile.agent_id
        and producer.version_sha256 == profile.agent_version_sha256
        and body.tool_name == profile.tool_name
        and body.tool_version_sha256 == profile.tool_version_sha256
        and body.fact_namespace == profile.fact_namespace
    )


def _profile_matches_claim(
    profile: ProvenanceProfileV1,
    payload: provenance.ProvenancePayloadV1,
) -> bool:
    body = payload.body
    return (
        type(body) is provenance.FactClaimBodyV1
        and payload.producer.kind is provenance.ProducerKindV1.USER
        and body.fact_namespace == profile.fact_namespace
    )


def _resolve_parent(
    *,
    child_evidence: PolicyEvidenceV2,
    child_payload: provenance.ProvenancePayloadV1,
    evidence_by_id: dict[str, PolicyEvidenceV2],
    payload_by_id: dict[str, provenance.ProvenancePayloadV1 | None],
    expected_kind: str,
) -> tuple[PolicyEvidenceV2, provenance.ProvenancePayloadV1] | None:
    if len(child_payload.links) != 1:
        return None
    link = child_payload.links[0]
    parent = evidence_by_id.get(link.target_evidence_id)
    if (
        parent is None
        or parent.evidence_content_sha256 != link.target_evidence_content_sha256
        or parent.kind != expected_kind
        or parent.ingest_order >= child_evidence.ingest_order
        or parent.session_id != child_evidence.session_id
        or parent.run_id != child_evidence.run_id
    ):
        return None
    parent_payload = payload_by_id.get(parent.evidence_id)
    if (
        parent_payload is None
        or parent_payload.trajectory_id != child_payload.trajectory_id
    ):
        return None
    return parent, parent_payload


def _member_from_evidence(value: PolicyEvidenceV2) -> EvidenceSnapshotMember:
    return EvidenceSnapshotMember(
        evidence_id=value.evidence_id,
        evidence_content_hash=value.evidence_content_sha256,
        ingest_order=value.ingest_order,
    )


def _resolve_provenance_evidence_graph_v1(
    *,
    profile: ProvenanceProfileV1,
    evidence: tuple[PolicyEvidenceV2, ...],
    learning_boundary_by_key: dict[str, int],
) -> ProvenanceGraphV1:
    """Resolve the current per-memory outcome epoch over the complete graph.

    The complete evidence snapshot remains addressable so that a newly emitted
    evaluator outcome can re-verify an older result (and its older ancestors).
    Only terminal outcomes newer than the target memory's exclusive boundary
    enter ambiguity resolution and consensus for this decision epoch.
    """

    profile = _validate_profile(profile)
    if (
        type(evidence) is not tuple
        or type(learning_boundary_by_key) is not dict
        or any(
            type(key) is not str
            or _KEY_PATTERN.fullmatch(key) is None
            or type(boundary) is not int
            or not -1 <= boundary <= _MAX_INT64
            for key, boundary in learning_boundary_by_key.items()
        )
    ):
        raise _error("closed_schema")
    evidence = tuple(_validate_evidence(item) for item in evidence)
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise _error("input_invariant")
    evidence_by_id = {item.evidence_id: item for item in evidence}
    payload_by_id = {
        item.evidence_id: _parse_evidence_payload(item) for item in evidence
    }
    all_outcome_items = tuple(
        item for item in evidence if item.kind == EvidenceKind.OUTCOME.value
    )
    outcome_items: list[PolicyEvidenceV2] = []
    profile_matching_outcomes_by_target: dict[str, list[PolicyEvidenceV2]] = {}
    for item in all_outcome_items:
        payload = payload_by_id[item.evidence_id]
        if payload is None or not _profile_matches_outcome(profile, payload):
            outcome_items.append(item)
            continue
        target_key: str | None = None
        if len(payload.links) == 1:
            target = evidence_by_id.get(payload.links[0].target_evidence_id)
            target_payload = (
                None if target is None else payload_by_id.get(target.evidence_id)
            )
            if (
                target_payload is not None
                and _profile_matches_result(profile, target_payload)
                and type(target_payload.body) is provenance.ToolResultBodyV1
            ):
                target_key = target_payload.body.key
        if (
            target_key is not None
            and item.ingest_order
            <= learning_boundary_by_key.get(target_key, -1)
        ):
            # Already closed by that memory's prior application.  It is not a
            # current rejection and cannot veto a later re-verification.
            continue
        outcome_items.append(item)
        if len(payload.links) == 1:
            # Deliberately aggregate before link resolution.  Any second
            # profile-matching declaration for the same content-addressed ID,
            # even one with a bad full hash or ordering, forces abstention in
            # this memory's current epoch.  Closed historical outcomes do not
            # poison a later explicit re-verification.
            profile_matching_outcomes_by_target.setdefault(
                payload.links[0].target_evidence_id,
                [],
            ).append(item)

    verified: list[VerifiedFactChainV1] = []
    accepted_outcome_ids: set[str] = set()
    for outcome in outcome_items:
        outcome_payload = payload_by_id[outcome.evidence_id]
        if (
            outcome_payload is None
            or not _profile_matches_outcome(profile, outcome_payload)
            or len(
                profile_matching_outcomes_by_target.get(
                    outcome_payload.links[0].target_evidence_id,
                    (),
                )
            )
            != 1
        ):
            continue
        outcome_body = outcome_payload.body
        if (
            type(outcome_body) is not provenance.VerificationOutcomeBodyV1
            or outcome_body.verdict is not provenance.VerificationVerdictV1.PASS
        ):
            continue
        resolved_result = _resolve_parent(
            child_evidence=outcome,
            child_payload=outcome_payload,
            evidence_by_id=evidence_by_id,
            payload_by_id=payload_by_id,
            expected_kind=EvidenceKind.TOOL_RESULT.value,
        )
        if resolved_result is None:
            continue
        result, result_payload = resolved_result
        if not _profile_matches_result(profile, result_payload):
            continue
        result_body = result_payload.body
        if (
            type(result_body) is not provenance.ToolResultBodyV1
            or result_body.status is not provenance.ToolResultStatusV1.OK
            or result_body.value is None
        ):
            continue
        resolved_call = _resolve_parent(
            child_evidence=result,
            child_payload=result_payload,
            evidence_by_id=evidence_by_id,
            payload_by_id=payload_by_id,
            expected_kind=EvidenceKind.TOOL_CALL.value,
        )
        if resolved_call is None:
            continue
        call, call_payload = resolved_call
        if not _profile_matches_call(profile, call_payload):
            continue
        call_body = call_payload.body
        if (
            type(call_body) is not provenance.ToolCallBodyV1
            or call_body.call_id != result_body.call_id
            or call_body.fact_namespace != result_body.fact_namespace
            or call_body.key != result_body.key
        ):
            continue
        resolved_claim = _resolve_parent(
            child_evidence=call,
            child_payload=call_payload,
            evidence_by_id=evidence_by_id,
            payload_by_id=payload_by_id,
            expected_kind=EvidenceKind.FEEDBACK.value,
        )
        if resolved_claim is None:
            # A user message is also a valid fact-claim role.
            resolved_claim = _resolve_parent(
                child_evidence=call,
                child_payload=call_payload,
                evidence_by_id=evidence_by_id,
                payload_by_id=payload_by_id,
                expected_kind=EvidenceKind.USER_MESSAGE.value,
            )
        if resolved_claim is None:
            continue
        claim, claim_payload = resolved_claim
        if not _profile_matches_claim(profile, claim_payload):
            continue
        claim_body = claim_payload.body
        if (
            type(claim_body) is not provenance.FactClaimBodyV1
            or claim_body.fact_namespace != call_body.fact_namespace
            or claim_body.key != call_body.key
        ):
            continue
        verified.append(
            VerifiedFactChainV1(
                fact_namespace=result_body.fact_namespace,
                key=result_body.key,
                value=result_body.value,
                call_id=result_body.call_id,
                trajectory_id=outcome_payload.trajectory_id,
                session_id=outcome.session_id,
                run_id=outcome.run_id,
                claim=_member_from_evidence(claim),
                tool_call=_member_from_evidence(call),
                tool_result=_member_from_evidence(result),
                outcome=_member_from_evidence(outcome),
            )
        )
        accepted_outcome_ids.add(outcome.evidence_id)

    chains_by_call: dict[
        tuple[str, str, str, str],
        list[VerifiedFactChainV1],
    ] = {}
    for chain in verified:
        call_identity = (
            chain.session_id,
            chain.run_id,
            chain.trajectory_id,
            chain.call_id,
        )
        chains_by_call.setdefault(call_identity, []).append(chain)
    ambiguous_call_identities = {
        identity for identity, chains in chains_by_call.items() if len(chains) != 1
    }
    if ambiguous_call_identities:
        accepted_outcome_ids.difference_update(
            chain.outcome.evidence_id
            for chain in verified
            if (
                chain.session_id,
                chain.run_id,
                chain.trajectory_id,
                chain.call_id,
            )
            in ambiguous_call_identities
        )
        verified = [
            chain
            for chain in verified
            if (
                chain.session_id,
                chain.run_id,
                chain.trajectory_id,
                chain.call_id,
            )
            not in ambiguous_call_identities
        ]

    ordered = tuple(
        sorted(
            verified,
            key=lambda item: (
                item.fact_namespace,
                item.key,
                item.outcome.ingest_order,
                item.outcome.evidence_id,
            ),
        )
    )
    rejected = tuple(
        sorted(
            item.evidence_id
            for item in outcome_items
            if item.evidence_id not in accepted_outcome_ids
        )
    )
    return ProvenanceGraphV1(
        verified_fact_chains=ordered,
        rejected_outcome_evidence_ids=rejected,
    )


def _resolve_provenance_graph_from_input_v1(
    value: PolicyInputV2,
) -> ProvenanceGraphV1:
    """Resolve chains from one structurally validated policy input."""

    value = _validate_policy_input(value)
    return _resolve_provenance_evidence_graph_v1(
        profile=value.provenance_profile,
        evidence=value.evidence,
        learning_boundary_by_key={
            base.key: base.learning_evidence_after_ingest_order
            for base in value.base_memories
        },
    )


def resolve_provenance_graph_v1(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    value: PolicyInputV2,
) -> ProvenanceGraphV1:
    """Reload the exact snapshot/base from SQLite, then resolve its graph.

    ``PolicyInputV2`` is intentionally constructible and its structural hash is
    not a store capability.  This public entry point authenticates it by
    rebuilding the exact input from ``store`` and rejecting any mismatch before
    following a link.
    """

    validated = _validate_policy_input(value)
    authentic = project_policy_input_from_snapshot_v2(
        store=store,
        scope=scope,
        base_release_id=validated.base_release_id,
        evidence_snapshot_id=validated.evidence_snapshot_id,
        provenance_profile=validated.provenance_profile,
    )
    if authentic != validated:
        raise _error("input_not_store_authentic")
    return _resolve_provenance_graph_from_input_v1(authentic)
