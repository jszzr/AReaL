# SPDX-License-Identifier: Apache-2.0

"""Immutable evidence snapshots for a trusted writer and drift detection.

The hashes are deterministic integrity commitments, not signatures against an
operator who can rewrite the database and recompute every commitment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from areal.v2.memory_service.types import (
    EvidenceKind,
    MemoryScope,
    _validate_aware_datetime,
    _validate_string,
)

EVIDENCE_SNAPSHOT_ORDERING_POLICY = "observed-at-sequence-evidence-id-v1"

_SCHEMA_VERSION = 1
_MAX_INGEST_ORDER = 2**63 - 1


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _scope_value(scope: MemoryScope) -> dict[str, str]:
    return {
        "tenant_id": scope.tenant_id,
        "namespace": scope.namespace,
        "subject_id": scope.subject_id,
    }


def _snapshot_allowed_kinds(value: object) -> tuple[EvidenceKind, ...]:
    if not isinstance(value, tuple):
        raise TypeError("allowed_kinds must be a tuple")
    snapshot = tuple(tuple.__iter__(value))
    if not snapshot:
        raise ValueError("allowed_kinds must not be empty")
    if any(type(item) is not EvidenceKind for item in snapshot):
        raise TypeError("allowed_kinds must contain only EvidenceKind values")
    if len(set(snapshot)) != len(snapshot):
        raise ValueError("allowed_kinds must not contain duplicates")
    return tuple(sorted(snapshot, key=lambda item: item.value))


def _snapshot_members(value: object) -> tuple[EvidenceSnapshotMember, ...]:
    if not isinstance(value, tuple):
        raise TypeError("members must be a tuple")
    snapshot = tuple(tuple.__iter__(value))
    if any(type(item) is not EvidenceSnapshotMember for item in snapshot):
        raise TypeError("members must contain only EvidenceSnapshotMember values")
    if len({item.evidence_id for item in snapshot}) != len(snapshot):
        raise ValueError("members must not contain duplicate evidence IDs")
    if len({item.ingest_order for item in snapshot}) != len(snapshot):
        raise ValueError("members must not contain duplicate ingest orders")
    return snapshot


@dataclass(frozen=True, slots=True)
class EvidenceSnapshotSpec:
    """The complete server-evaluated predicate for one evidence snapshot."""

    scope: MemoryScope
    allowed_kinds: tuple[EvidenceKind, ...]
    cutoff: datetime

    def __post_init__(self) -> None:
        if type(self.scope) is not MemoryScope:
            raise TypeError("scope must be a MemoryScope")
        allowed_kinds = _snapshot_allowed_kinds(self.allowed_kinds)
        cutoff = _validate_aware_datetime(self.cutoff, "cutoff")
        object.__setattr__(self, "allowed_kinds", allowed_kinds)
        object.__setattr__(self, "cutoff", cutoff)

    def canonical_bytes(self) -> bytes:
        """Serialize the selection predicate as deterministic UTF-8 JSON."""

        return _canonical_json_bytes(
            {
                "allowed_kinds": [kind.value for kind in self.allowed_kinds],
                "cutoff_utc": self.cutoff.isoformat(),
                "schema_version": _SCHEMA_VERSION,
                "scope": _scope_value(self.scope),
            }
        )


@dataclass(frozen=True, slots=True)
class EvidenceSnapshotMember:
    """One ordered evidence commitment stored in a snapshot."""

    evidence_id: str
    evidence_content_hash: str
    ingest_order: int

    def __post_init__(self) -> None:
        evidence_id = _validate_string(self.evidence_id, "evidence_id")
        evidence_content_hash = _validate_string(
            self.evidence_content_hash,
            "evidence_content_hash",
        )
        if type(self.ingest_order) is not int:
            raise TypeError("ingest_order must be an integer")
        if self.ingest_order < 0 or self.ingest_order > _MAX_INGEST_ORDER:
            raise ValueError("ingest_order must fit the non-negative signed-64 range")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "evidence_content_hash", evidence_content_hash)


def _snapshot_canonical_bytes(
    *,
    spec: EvidenceSnapshotSpec,
    evidence_high_watermark: int,
    ordering_policy: str,
    members: tuple[EvidenceSnapshotMember, ...],
) -> bytes:
    return _canonical_json_bytes(
        {
            "allowed_kinds": [kind.value for kind in spec.allowed_kinds],
            "cutoff_utc": spec.cutoff.isoformat(),
            "evidence_high_watermark": evidence_high_watermark,
            "members": [
                {
                    "evidence_content_hash": member.evidence_content_hash,
                    "evidence_id": member.evidence_id,
                    "ingest_order": member.ingest_order,
                }
                for member in members
            ],
            "ordering_policy": ordering_policy,
            "schema_version": _SCHEMA_VERSION,
            "scope": _scope_value(spec.scope),
        }
    )


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """An immutable, high-watermark-bounded commitment to complete evidence."""

    snapshot_id: str
    spec: EvidenceSnapshotSpec
    evidence_high_watermark: int
    ordering_policy: str
    members: tuple[EvidenceSnapshotMember, ...]
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        snapshot_id = _validate_string(self.snapshot_id, "snapshot_id")
        if type(self.spec) is not EvidenceSnapshotSpec:
            raise TypeError("spec must be an EvidenceSnapshotSpec")
        if type(self.evidence_high_watermark) is not int:
            raise TypeError("evidence_high_watermark must be an integer")
        if not -1 <= self.evidence_high_watermark <= _MAX_INGEST_ORDER:
            raise ValueError(
                "evidence_high_watermark must fit the signed-64 range from -1"
            )
        ordering_policy = _validate_string(self.ordering_policy, "ordering_policy")
        if ordering_policy != EVIDENCE_SNAPSHOT_ORDERING_POLICY:
            raise ValueError("ordering_policy is not supported")
        members = _snapshot_members(self.members)
        if any(
            member.ingest_order > self.evidence_high_watermark for member in members
        ):
            raise ValueError("members must not exceed evidence_high_watermark")
        content_hash = _validate_string(self.content_hash, "content_hash")
        created_at = _validate_aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "ordering_policy", ordering_policy)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "created_at", created_at)

    def canonical_bytes(self) -> bytes:
        """Serialize every selection and membership commitment deterministically."""

        return _snapshot_canonical_bytes(
            spec=self.spec,
            evidence_high_watermark=self.evidence_high_watermark,
            ordering_policy=self.ordering_policy,
            members=self.members,
        )
