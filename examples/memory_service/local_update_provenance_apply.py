# SPDX-License-Identifier: Apache-2.0

"""Validate and atomically publish one provenance-grounded Memory decision.

The semantic validator reads the complete immutable source snapshot.  The core
SQLite commit then uses that snapshot's global ingest order as a boundary while
holding ``BEGIN IMMEDIATE`` and rejects any later evidence in the same scope.
It writes candidates, revisions, the result release, and its application ledger
in the same transaction.  A relevant interleaving therefore causes a zero-write
stale failure instead of being mislabeled as post-release evidence.
"""

from __future__ import annotations

import re

from examples.memory_service import local_update_provenance_projection as projection

from areal.v2.memory_service import (
    MemoryApplicationProposal,
    MemoryApplicationStaleSnapshotError,
    MemoryApplicationUpdateProposal,
    MemoryApplicationV1,
    MemoryScope,
    RevisionOperation,
)
from areal.v2.memory_service.errors import MemoryServiceError
from areal.v2.memory_service.sqlite_store import SQLiteMemoryStore

__all__ = [
    "LocalUpdateProvenanceApplyError",
    "VERIFIED_CHAIN_POLICY_ID_V1",
    "VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1",
    "commit_verified_chain_application_v1",
    "verified_chain_decision_sha256_v1",
]


class LocalUpdateProvenanceApplyError(RuntimeError):
    """Stable, payload-free reason for rejecting an application."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("apply error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


VERIFIED_CHAIN_POLICY_ID_V1 = projection.VERIFIED_CHAIN_POLICY_ID_V1
VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1 = (
    projection.VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1
)

_FACT_PATTERN = re.compile(
    r"(?P<key>project-[abcdefghjklmnpqrstuvwxyz23456789]{6}) = "
    r"(?P<value>[ABCDEFGHJKLMNPQRSTUVWXYZ23456789]{5})"
)
def _updates_value(
    updates: object,
) -> tuple[MemoryApplicationUpdateProposal, ...]:
    if (
        type(updates) is not tuple
        or not updates
        or any(type(item) is not MemoryApplicationUpdateProposal for item in updates)
    ):
        raise LocalUpdateProvenanceApplyError("decision_invalid")
    return updates


def verified_chain_decision_sha256_v1(
    source_input: projection.PolicyInputV2,
    updates: tuple[MemoryApplicationUpdateProposal, ...],
) -> str:
    """Hash the exact input and ordered changed updates of one decision."""

    try:
        projection.policy_input_sha256_v2(source_input)
    except (projection.LocalUpdateProvenanceProjectionError, TypeError, ValueError):
        raise LocalUpdateProvenanceApplyError("input_invalid") from None
    updates = _updates_value(updates)
    try:
        return projection.verified_chain_decision_sha256_v1(source_input, updates)
    except projection.LocalUpdateProvenanceProjectionError:
        raise LocalUpdateProvenanceApplyError("decision_invalid") from None


def _validate_updates(
    *,
    source_input: projection.PolicyInputV2,
    graph: projection.ProvenanceGraphV1,
    updates: tuple[MemoryApplicationUpdateProposal, ...],
) -> None:
    base_by_revision = {base.revision_id: base for base in source_input.base_memories}
    base_by_key = {base.key: base for base in source_input.base_memories}
    seen_keys: set[str] = set()
    for update in updates:
        match = _FACT_PATTERN.fullmatch(update.content)
        if match is None:
            raise LocalUpdateProvenanceApplyError("decision_invalid")
        key = match.group("key")
        fact_value = match.group("value")
        grounding = projection.canonical_verified_fact_grounding_v1(
            profile=source_input.provenance_profile,
            graph=graph,
            key=key,
            fact_value=fact_value,
        )
        if (
            not grounding
            or update.evidence_ids != tuple(member.evidence_id for member in grounding)
            or key in seen_keys
        ):
            raise LocalUpdateProvenanceApplyError("decision_invalid")
        seen_keys.add(key)
        if update.operation is RevisionOperation.SUPERSEDE:
            assert update.parent_revision_id is not None
            parent = base_by_revision.get(update.parent_revision_id)
            if parent is None or parent.key != key or parent.value == fact_value:
                raise LocalUpdateProvenanceApplyError("decision_invalid")
        elif update.operation is RevisionOperation.ADD:
            if key in base_by_key:
                raise LocalUpdateProvenanceApplyError("decision_invalid")
        else:
            raise LocalUpdateProvenanceApplyError("decision_invalid")


def commit_verified_chain_application_v1(
    *,
    store: SQLiteMemoryStore,
    scope: MemoryScope,
    source_input: projection.PolicyInputV2,
    updates: tuple[MemoryApplicationUpdateProposal, ...],
    idempotency_key: str,
) -> MemoryApplicationV1:
    """Validate the complete graph and atomically publish one changed release."""

    if (
        type(store) is not SQLiteMemoryStore
        or type(scope) is not MemoryScope
        or type(idempotency_key) is not str
        or not idempotency_key.strip()
    ):
        raise LocalUpdateProvenanceApplyError("closed_schema")
    try:
        idempotency_key.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise LocalUpdateProvenanceApplyError("closed_schema") from None
    updates = _updates_value(updates)
    try:
        graph = projection.resolve_provenance_graph_v1(
            store=store,
            scope=scope,
            value=source_input,
        )
        _validate_updates(
            source_input=source_input,
            graph=graph,
            updates=updates,
        )
        decision_sha256 = verified_chain_decision_sha256_v1(
            source_input,
            updates,
        )
        proposal = MemoryApplicationProposal(
            scope=scope,
            source_snapshot_id=source_input.evidence_snapshot_id,
            source_base_release_id=source_input.base_release_id,
            projector_id=projection.PROVENANCE_PROJECTOR_ID_V1,
            projector_version_sha256=(
                projection.PROVENANCE_PROJECTOR_VERSION_SHA256_V1
            ),
            policy_id=VERIFIED_CHAIN_POLICY_ID_V1,
            policy_version_sha256=VERIFIED_CHAIN_POLICY_VERSION_SHA256_V1,
            policy_input_sha256=projection.policy_input_sha256_v2(source_input),
            decision_sha256=decision_sha256,
            policy_context=projection.provenance_application_context_v1(
                source_input.provenance_profile
            ),
            updates=updates,
            idempotency_key=idempotency_key,
        )
        return store.commit_memory_application(proposal)
    except LocalUpdateProvenanceApplyError:
        raise
    except MemoryApplicationStaleSnapshotError:
        raise LocalUpdateProvenanceApplyError("source_snapshot_stale") from None
    except projection.LocalUpdateProvenanceProjectionError:
        raise LocalUpdateProvenanceApplyError("input_invalid") from None
    except (MemoryServiceError, OverflowError, TypeError, ValueError):
        raise LocalUpdateProvenanceApplyError("commit_failed") from None
