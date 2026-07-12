# SPDX-License-Identifier: Apache-2.0

"""Immutable commitments for one atomic memory-policy application.

An application is the durable join between a sealed policy input and the
candidate, revision, and release objects produced from that input.  These
values are integrity commitments made by the trusted Memory Service writer;
they are not signatures against an operator who can rewrite the database and
recompute every hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import NoReturn

from areal.v2.memory_service.history_types import RevisionOperation
from areal.v2.memory_service.snapshot_types import EvidenceSnapshotMember
from areal.v2.memory_service.types import MemoryScope

_SCHEMA_VERSION = 1
_MAX_SIGNED_64 = 2**63 - 1
_LOWER_HEX = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_string(
    value: object,
    field_name: str,
    *,
    allow_blank: bool = False,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not allow_blank and not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc
    return value


def _sha256_string(value: object, field_name: str) -> str:
    value = _strict_string(value, field_name)
    if len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _bounded_integer(
    value: object,
    field_name: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum or value > _MAX_SIGNED_64:
        raise ValueError(f"{field_name} must fit the signed-64 range from {minimum}")
    return value


def _aware_datetime(value: object, field_name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be normalizable to UTC") from exc
    if offset is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be normalizable to UTC") from exc


def _scope_value(value: object) -> dict[str, str]:
    if type(value) is not MemoryScope:
        raise TypeError("scope must be a MemoryScope")
    return {
        "namespace": _strict_string(value.namespace, "scope.namespace"),
        "subject_id": _strict_string(value.subject_id, "scope.subject_id"),
        "tenant_id": _strict_string(value.tenant_id, "scope.tenant_id"),
    }


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"policy_context must not contain {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("policy_context must not contain duplicate object keys")
        result[key] = value
    return result


def _canonical_json_string(value: object, field_name: str) -> str:
    value = _strict_string(value, field_name, allow_blank=True)
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    try:
        canonical = _canonical_json_bytes(decoded).decode("utf-8")
    except (OverflowError, TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain canonicalizable JSON") from exc
    if value != canonical:
        raise ValueError(
            f"{field_name} must be compact canonical JSON with sorted object keys"
        )
    return value


def _strict_tuple(value: object, field_name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    return value


def _string_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    values = _strict_tuple(value, field_name)
    snapshot = tuple(
        _strict_string(item, field_name) for item in tuple.__iter__(values)
    )
    if not allow_empty and not snapshot:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(snapshot)) != len(snapshot):
        raise ValueError(f"{field_name} must not contain duplicates")
    return snapshot


def _member_value(value: object) -> dict[str, object]:
    if type(value) is not EvidenceSnapshotMember:
        raise TypeError("grounding must contain only EvidenceSnapshotMember values")
    return {
        "evidence_content_hash": _sha256_string(
            value.evidence_content_hash,
            "grounding.evidence_content_hash",
        ),
        "evidence_id": _strict_string(value.evidence_id, "grounding.evidence_id"),
        "ingest_order": _bounded_integer(
            value.ingest_order,
            "grounding.ingest_order",
        ),
    }


def _grounding_values(value: object) -> tuple[dict[str, object], ...]:
    values = _strict_tuple(value, "grounding")
    if not values:
        raise ValueError("grounding must not be empty")
    snapshot = tuple(_member_value(member) for member in tuple.__iter__(values))
    evidence_ids = tuple(item["evidence_id"] for item in snapshot)
    ingest_orders = tuple(item["ingest_order"] for item in snapshot)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("grounding must not contain duplicate evidence IDs")
    if len(set(ingest_orders)) != len(ingest_orders):
        raise ValueError("grounding must not contain duplicate ingest orders")
    return snapshot


def _operation_and_parent(
    operation: object,
    parent_revision_id: object,
) -> tuple[RevisionOperation, str | None]:
    if type(operation) is not RevisionOperation:
        raise TypeError("operation must be a RevisionOperation")
    if operation not in {RevisionOperation.ADD, RevisionOperation.SUPERSEDE}:
        raise ValueError("operation must be ADD or SUPERSEDE")
    if parent_revision_id is not None:
        parent_revision_id = _strict_string(
            parent_revision_id,
            "parent_revision_id",
        )
    if operation is RevisionOperation.ADD and parent_revision_id is not None:
        raise ValueError("parent_revision_id must be absent for ADD")
    if operation is RevisionOperation.SUPERSEDE and parent_revision_id is None:
        raise ValueError("parent_revision_id is required for SUPERSEDE")
    return operation, parent_revision_id


@dataclass(frozen=True, slots=True)
class MemoryApplicationUpdateProposal:
    """One proposed ADD or SUPERSEDE in an atomic policy decision."""

    content: str
    evidence_ids: tuple[str, ...]
    operation: RevisionOperation
    parent_revision_id: str | None

    def __post_init__(self) -> None:
        content = _strict_string(self.content, "content")
        evidence_ids = _string_tuple(
            self.evidence_ids,
            "evidence_ids",
            allow_empty=False,
        )
        operation, parent_revision_id = _operation_and_parent(
            self.operation,
            self.parent_revision_id,
        )
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parent_revision_id", parent_revision_id)

    def _canonical_value(self) -> dict[str, object]:
        content = _strict_string(self.content, "content")
        evidence_ids = _string_tuple(
            self.evidence_ids,
            "evidence_ids",
            allow_empty=False,
        )
        operation, parent_revision_id = _operation_and_parent(
            self.operation,
            self.parent_revision_id,
        )
        return {
            "content": content,
            "evidence_ids": list(evidence_ids),
            "operation": operation.value,
            "parent_revision_id": parent_revision_id,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                **self._canonical_value(),
            }
        )


def _update_value(value: object) -> dict[str, object]:
    if type(value) is not MemoryApplicationUpdateProposal:
        raise TypeError(
            "updates must contain only MemoryApplicationUpdateProposal values"
        )
    return value._canonical_value()


def _updates_values(value: object) -> tuple[dict[str, object], ...]:
    values = _strict_tuple(value, "updates")
    if not values:
        raise ValueError("updates must not be empty")
    snapshot = tuple(_update_value(update) for update in tuple.__iter__(values))
    canonical_updates = tuple(_canonical_json_bytes(item) for item in snapshot)
    if len(set(canonical_updates)) != len(canonical_updates):
        raise ValueError("updates must not contain duplicates")
    parent_revision_ids = tuple(
        item["parent_revision_id"]
        for item in snapshot
        if item["parent_revision_id"] is not None
    )
    if len(set(parent_revision_ids)) != len(parent_revision_ids):
        raise ValueError("updates must not supersede the same parent more than once")
    return snapshot


@dataclass(frozen=True, slots=True)
class MemoryApplicationProposal:
    """The complete, idempotent input to one atomic memory application."""

    scope: MemoryScope
    source_snapshot_id: str
    source_base_release_id: str
    projector_id: str
    projector_version_sha256: str
    policy_id: str
    policy_version_sha256: str
    policy_input_sha256: str
    decision_sha256: str
    policy_context: str
    updates: tuple[MemoryApplicationUpdateProposal, ...]
    idempotency_key: str

    def __post_init__(self) -> None:
        _scope_value(self.scope)
        source_snapshot_id = _strict_string(
            self.source_snapshot_id,
            "source_snapshot_id",
        )
        source_base_release_id = _strict_string(
            self.source_base_release_id,
            "source_base_release_id",
        )
        projector_id = _strict_string(self.projector_id, "projector_id")
        projector_version_sha256 = _sha256_string(
            self.projector_version_sha256,
            "projector_version_sha256",
        )
        policy_id = _strict_string(self.policy_id, "policy_id")
        policy_version_sha256 = _sha256_string(
            self.policy_version_sha256,
            "policy_version_sha256",
        )
        policy_input_sha256 = _sha256_string(
            self.policy_input_sha256,
            "policy_input_sha256",
        )
        decision_sha256 = _sha256_string(
            self.decision_sha256,
            "decision_sha256",
        )
        policy_context = _canonical_json_string(
            self.policy_context,
            "policy_context",
        )
        updates = self.updates
        _updates_values(updates)
        idempotency_key = _strict_string(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "source_snapshot_id", source_snapshot_id)
        object.__setattr__(
            self,
            "source_base_release_id",
            source_base_release_id,
        )
        object.__setattr__(self, "projector_id", projector_id)
        object.__setattr__(
            self,
            "projector_version_sha256",
            projector_version_sha256,
        )
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(
            self,
            "policy_version_sha256",
            policy_version_sha256,
        )
        object.__setattr__(self, "policy_input_sha256", policy_input_sha256)
        object.__setattr__(self, "decision_sha256", decision_sha256)
        object.__setattr__(self, "policy_context", policy_context)
        object.__setattr__(self, "idempotency_key", idempotency_key)

    def _canonical_value(self) -> dict[str, object]:
        return {
            "decision_sha256": _sha256_string(
                self.decision_sha256,
                "decision_sha256",
            ),
            "idempotency_key": _strict_string(
                self.idempotency_key,
                "idempotency_key",
            ),
            "policy_context": _canonical_json_string(
                self.policy_context,
                "policy_context",
            ),
            "policy_id": _strict_string(self.policy_id, "policy_id"),
            "policy_input_sha256": _sha256_string(
                self.policy_input_sha256,
                "policy_input_sha256",
            ),
            "policy_version_sha256": _sha256_string(
                self.policy_version_sha256,
                "policy_version_sha256",
            ),
            "projector_id": _strict_string(self.projector_id, "projector_id"),
            "projector_version_sha256": _sha256_string(
                self.projector_version_sha256,
                "projector_version_sha256",
            ),
            "scope": _scope_value(self.scope),
            "source_base_release_id": _strict_string(
                self.source_base_release_id,
                "source_base_release_id",
            ),
            "source_snapshot_id": _strict_string(
                self.source_snapshot_id,
                "source_snapshot_id",
            ),
            "updates": list(_updates_values(self.updates)),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                **self._canonical_value(),
            }
        )


@dataclass(frozen=True, slots=True)
class AppliedMemoryUpdateV1:
    """The exact stored objects created for one proposed update."""

    ordinal: int
    release_position: int
    operation: RevisionOperation
    grounding: tuple[EvidenceSnapshotMember, ...]
    candidate_id: str
    candidate_content_sha256: str
    revision_id: str
    revision_content_sha256: str
    memory_id: str
    generation: int
    parent_revision_id: str | None

    def __post_init__(self) -> None:
        _bounded_integer(self.ordinal, "ordinal")
        _bounded_integer(self.release_position, "release_position")
        operation, parent_revision_id = _operation_and_parent(
            self.operation,
            self.parent_revision_id,
        )
        _grounding_values(self.grounding)
        candidate_id = _strict_string(self.candidate_id, "candidate_id")
        candidate_content_sha256 = _sha256_string(
            self.candidate_content_sha256,
            "candidate_content_sha256",
        )
        revision_id = _strict_string(self.revision_id, "revision_id")
        revision_content_sha256 = _sha256_string(
            self.revision_content_sha256,
            "revision_content_sha256",
        )
        memory_id = _strict_string(self.memory_id, "memory_id")
        _bounded_integer(self.generation, "generation")
        if operation is RevisionOperation.ADD and self.generation != 0:
            raise ValueError("ADD generation must be zero")
        if operation is RevisionOperation.SUPERSEDE and self.generation == 0:
            raise ValueError("SUPERSEDE generation must be positive")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parent_revision_id", parent_revision_id)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(
            self,
            "candidate_content_sha256",
            candidate_content_sha256,
        )
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(
            self,
            "revision_content_sha256",
            revision_content_sha256,
        )
        object.__setattr__(self, "memory_id", memory_id)

    def _canonical_value(self) -> dict[str, object]:
        operation, parent_revision_id = _operation_and_parent(
            self.operation,
            self.parent_revision_id,
        )
        generation = _bounded_integer(self.generation, "generation")
        if operation is RevisionOperation.ADD and generation != 0:
            raise ValueError("ADD generation must be zero")
        if operation is RevisionOperation.SUPERSEDE and generation == 0:
            raise ValueError("SUPERSEDE generation must be positive")
        return {
            "candidate_content_sha256": _sha256_string(
                self.candidate_content_sha256,
                "candidate_content_sha256",
            ),
            "candidate_id": _strict_string(self.candidate_id, "candidate_id"),
            "generation": generation,
            "grounding": list(_grounding_values(self.grounding)),
            "memory_id": _strict_string(self.memory_id, "memory_id"),
            "operation": operation.value,
            "ordinal": _bounded_integer(self.ordinal, "ordinal"),
            "parent_revision_id": parent_revision_id,
            "release_position": _bounded_integer(
                self.release_position,
                "release_position",
            ),
            "revision_content_sha256": _sha256_string(
                self.revision_content_sha256,
                "revision_content_sha256",
            ),
            "revision_id": _strict_string(self.revision_id, "revision_id"),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "schema_version": _SCHEMA_VERSION,
                **self._canonical_value(),
            }
        )


def _applied_value(value: object) -> dict[str, object]:
    if type(value) is not AppliedMemoryUpdateV1:
        raise TypeError(
            "applied_updates must contain only AppliedMemoryUpdateV1 values"
        )
    return value._canonical_value()


def _applied_values(value: object) -> tuple[dict[str, object], ...]:
    values = _strict_tuple(value, "applied_updates")
    if not values:
        raise ValueError("applied_updates must not be empty")
    return tuple(_applied_value(item) for item in tuple.__iter__(values))


def _application_canonical_bytes(
    *,
    proposal: object,
    source_snapshot_content_sha256: object,
    source_evidence_high_watermark: object,
    base_release_content_sha256: object,
    result_release_id: object,
    result_release_content_sha256: object,
    result_revision_ids: object,
    applied_updates: object,
    application_order: object,
) -> bytes:
    if type(proposal) is not MemoryApplicationProposal:
        raise TypeError("proposal must be a MemoryApplicationProposal")
    proposal_value = proposal._canonical_value()
    source_snapshot_content_sha256 = _sha256_string(
        source_snapshot_content_sha256,
        "source_snapshot_content_sha256",
    )
    source_evidence_high_watermark = _bounded_integer(
        source_evidence_high_watermark,
        "source_evidence_high_watermark",
        minimum=-1,
    )
    base_release_content_sha256 = _sha256_string(
        base_release_content_sha256,
        "base_release_content_sha256",
    )
    result_release_id = _strict_string(result_release_id, "result_release_id")
    result_release_content_sha256 = _sha256_string(
        result_release_content_sha256,
        "result_release_content_sha256",
    )
    result_revision_ids = _string_tuple(
        result_revision_ids,
        "result_revision_ids",
        allow_empty=False,
    )
    applied_update_values = _applied_values(applied_updates)
    application_order = _bounded_integer(application_order, "application_order")

    proposal_updates = proposal_value["updates"]
    assert type(proposal_updates) is list
    if len(applied_update_values) != len(proposal_updates):
        raise ValueError("applied_updates must correspond one-to-one with updates")
    if tuple(item["ordinal"] for item in applied_update_values) != tuple(
        range(len(applied_update_values))
    ):
        raise ValueError("applied_updates must be ordered by contiguous ordinal")

    release_positions = tuple(
        item["release_position"] for item in applied_update_values
    )
    if len(set(release_positions)) != len(release_positions):
        raise ValueError("applied_updates must not repeat a release position")
    if any(position >= len(result_revision_ids) for position in release_positions):
        raise ValueError("release_position must address result_revision_ids")

    candidate_ids = tuple(item["candidate_id"] for item in applied_update_values)
    revision_ids = tuple(item["revision_id"] for item in applied_update_values)
    memory_ids = tuple(item["memory_id"] for item in applied_update_values)
    for values, field_name in (
        (candidate_ids, "candidate IDs"),
        (revision_ids, "revision IDs"),
        (memory_ids, "memory IDs"),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"applied_updates must not repeat {field_name}")

    for proposal_update, applied_update in zip(
        proposal_updates,
        applied_update_values,
        strict=True,
    ):
        if applied_update["operation"] != proposal_update["operation"]:
            raise ValueError("applied operation drifted from the proposed update")
        if (
            applied_update["parent_revision_id"]
            != proposal_update["parent_revision_id"]
        ):
            raise ValueError("applied parent drifted from the proposed update")
        grounding_ids = [
            member["evidence_id"] for member in applied_update["grounding"]
        ]
        if grounding_ids != proposal_update["evidence_ids"]:
            raise ValueError("applied grounding drifted from proposed evidence_ids")
        release_position = applied_update["release_position"]
        if result_revision_ids[release_position] != applied_update["revision_id"]:
            raise ValueError("result release membership drifted from applied revision")
        if any(
            member["ingest_order"] > source_evidence_high_watermark
            for member in applied_update["grounding"]
        ):
            raise ValueError("applied grounding exceeds source_evidence_high_watermark")

    return _canonical_json_bytes(
        {
            "application_order": application_order,
            "applied_updates": list(applied_update_values),
            "base_release_content_sha256": base_release_content_sha256,
            "proposal": proposal_value,
            "result_release_content_sha256": result_release_content_sha256,
            "result_release_id": result_release_id,
            "result_revision_ids": list(result_revision_ids),
            "schema_version": _SCHEMA_VERSION,
            "source_evidence_high_watermark": source_evidence_high_watermark,
            "source_snapshot_content_sha256": source_snapshot_content_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryApplicationV1:
    """A content-addressed, atomic application stored by the Memory Service.

    ``application_order`` is intentionally part of the canonical commitment so
    a stored lineage edge cannot be replayed at another ledger position.
    ``created_at`` is storage metadata, matching other Memory Service records,
    and therefore is not part of the content address.
    """

    proposal: MemoryApplicationProposal
    source_snapshot_content_sha256: str
    source_evidence_high_watermark: int
    base_release_content_sha256: str
    result_release_id: str
    result_release_content_sha256: str
    result_revision_ids: tuple[str, ...]
    applied_updates: tuple[AppliedMemoryUpdateV1, ...]
    application_order: int
    application_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _application_canonical_bytes(
            proposal=self.proposal,
            source_snapshot_content_sha256=self.source_snapshot_content_sha256,
            source_evidence_high_watermark=self.source_evidence_high_watermark,
            base_release_content_sha256=self.base_release_content_sha256,
            result_release_id=self.result_release_id,
            result_release_content_sha256=self.result_release_content_sha256,
            result_revision_ids=self.result_revision_ids,
            applied_updates=self.applied_updates,
            application_order=self.application_order,
        )
        application_id = _strict_string(self.application_id, "application_id")
        content_hash = _sha256_string(self.content_hash, "content_hash")
        expected_content_hash = sha256(canonical).hexdigest()
        expected_application_id = f"mapp_{expected_content_hash[:24]}"
        if content_hash != expected_content_hash:
            raise ValueError("content_hash disagrees with canonical application bytes")
        if application_id != expected_application_id:
            raise ValueError(
                "application_id disagrees with canonical application bytes"
            )
        created_at = _aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "application_id", application_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "created_at", created_at)

    @classmethod
    def create(
        cls,
        *,
        proposal: MemoryApplicationProposal,
        source_snapshot_content_sha256: str,
        source_evidence_high_watermark: int,
        base_release_content_sha256: str,
        result_release_id: str,
        result_release_content_sha256: str,
        result_revision_ids: tuple[str, ...],
        applied_updates: tuple[AppliedMemoryUpdateV1, ...],
        application_order: int,
        created_at: datetime | None = None,
    ) -> MemoryApplicationV1:
        """Create and content-address an application from committed fields."""

        canonical = _application_canonical_bytes(
            proposal=proposal,
            source_snapshot_content_sha256=source_snapshot_content_sha256,
            source_evidence_high_watermark=source_evidence_high_watermark,
            base_release_content_sha256=base_release_content_sha256,
            result_release_id=result_release_id,
            result_release_content_sha256=result_release_content_sha256,
            result_revision_ids=result_revision_ids,
            applied_updates=applied_updates,
            application_order=application_order,
        )
        content_hash = sha256(canonical).hexdigest()
        if created_at is None:
            created_at = datetime.now(UTC)
        return cls(
            proposal=proposal,
            source_snapshot_content_sha256=source_snapshot_content_sha256,
            source_evidence_high_watermark=source_evidence_high_watermark,
            base_release_content_sha256=base_release_content_sha256,
            result_release_id=result_release_id,
            result_release_content_sha256=result_release_content_sha256,
            result_revision_ids=result_revision_ids,
            applied_updates=applied_updates,
            application_order=application_order,
            application_id=f"mapp_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=created_at,
        )

    def canonical_bytes(self) -> bytes:
        """Return canonical bytes and fail closed if the record has drifted."""

        canonical = _application_canonical_bytes(
            proposal=self.proposal,
            source_snapshot_content_sha256=self.source_snapshot_content_sha256,
            source_evidence_high_watermark=self.source_evidence_high_watermark,
            base_release_content_sha256=self.base_release_content_sha256,
            result_release_id=self.result_release_id,
            result_release_content_sha256=self.result_release_content_sha256,
            result_revision_ids=self.result_revision_ids,
            applied_updates=self.applied_updates,
            application_order=self.application_order,
        )
        expected_content_hash = sha256(canonical).hexdigest()
        if self.content_hash != expected_content_hash:
            raise ValueError("content_hash disagrees with canonical application bytes")
        if self.application_id != f"mapp_{expected_content_hash[:24]}":
            raise ValueError(
                "application_id disagrees with canonical application bytes"
            )
        return canonical


def _root_canonical_bytes(
    *,
    scope: object,
    release_id: object,
    release_content_sha256: object,
) -> bytes:
    return _canonical_json_bytes(
        {
            "release_content_sha256": _sha256_string(
                release_content_sha256,
                "release_content_sha256",
            ),
            "release_id": _strict_string(release_id, "release_id"),
            "schema_version": _SCHEMA_VERSION,
            "scope": _scope_value(scope),
        }
    )


@dataclass(frozen=True, slots=True)
class MemoryApplicationRootV1:
    """An explicit trust root for a release not produced by an application."""

    scope: MemoryScope
    release_id: str
    release_content_sha256: str
    root_id: str
    content_hash: str
    created_at: datetime

    def __post_init__(self) -> None:
        canonical = _root_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
        )
        root_id = _strict_string(self.root_id, "root_id")
        content_hash = _sha256_string(self.content_hash, "content_hash")
        expected_content_hash = sha256(canonical).hexdigest()
        if content_hash != expected_content_hash:
            raise ValueError("content_hash disagrees with canonical root bytes")
        if root_id != f"mroot_{expected_content_hash[:24]}":
            raise ValueError("root_id disagrees with canonical root bytes")
        created_at = _aware_datetime(self.created_at, "created_at")
        object.__setattr__(self, "root_id", root_id)
        object.__setattr__(self, "content_hash", content_hash)
        object.__setattr__(self, "created_at", created_at)

    @classmethod
    def create(
        cls,
        *,
        scope: MemoryScope,
        release_id: str,
        release_content_sha256: str,
        created_at: datetime | None = None,
    ) -> MemoryApplicationRootV1:
        canonical = _root_canonical_bytes(
            scope=scope,
            release_id=release_id,
            release_content_sha256=release_content_sha256,
        )
        content_hash = sha256(canonical).hexdigest()
        if created_at is None:
            created_at = datetime.now(UTC)
        return cls(
            scope=scope,
            release_id=release_id,
            release_content_sha256=release_content_sha256,
            root_id=f"mroot_{content_hash[:24]}",
            content_hash=content_hash,
            created_at=created_at,
        )

    def canonical_bytes(self) -> bytes:
        canonical = _root_canonical_bytes(
            scope=self.scope,
            release_id=self.release_id,
            release_content_sha256=self.release_content_sha256,
        )
        expected_content_hash = sha256(canonical).hexdigest()
        if self.content_hash != expected_content_hash:
            raise ValueError("content_hash disagrees with canonical root bytes")
        if self.root_id != f"mroot_{expected_content_hash[:24]}":
            raise ValueError("root_id disagrees with canonical root bytes")
        return canonical
