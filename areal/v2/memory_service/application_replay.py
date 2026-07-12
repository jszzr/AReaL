# SPDX-License-Identifier: Apache-2.0

"""Consistent, integrity-validated views of one Memory application lineage.

The view adds no signature or attestation.  Its trust comes from the SQLite
loader validating every durable commitment in one read transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from areal.v2.memory_service.application_types import (
    MemoryApplicationRootV1,
    MemoryApplicationV1,
)
from areal.v2.memory_service.history_types import MemoryRevision
from areal.v2.memory_service.release_types import MemoryRelease
from areal.v2.memory_service.snapshot_types import EvidenceSnapshot
from areal.v2.memory_service.types import EvidenceRecord


@dataclass(frozen=True, slots=True)
class MemoryApplicationReplayStepV1:
    """One application with every immutable object needed to replay it."""

    application: MemoryApplicationV1
    source_snapshot: EvidenceSnapshot
    source_evidence: tuple[EvidenceRecord, ...]
    result_release: MemoryRelease
    result_revisions: tuple[MemoryRevision, ...]

    def __post_init__(self) -> None:
        if (
            type(self.application) is not MemoryApplicationV1
            or type(self.source_snapshot) is not EvidenceSnapshot
            or type(self.source_evidence) is not tuple
            or any(type(item) is not EvidenceRecord for item in self.source_evidence)
            or type(self.result_release) is not MemoryRelease
            or type(self.result_revisions) is not tuple
            or any(type(item) is not MemoryRevision for item in self.result_revisions)
        ):
            raise TypeError("replay step values must use exact Memory Service types")
        application = self.application
        scope = application.proposal.scope
        snapshot = self.source_snapshot
        if (
            snapshot.spec.scope != scope
            or application.proposal.source_snapshot_id != snapshot.snapshot_id
            or application.source_snapshot_content_sha256 != snapshot.content_hash
            or application.source_evidence_high_watermark
            != snapshot.evidence_high_watermark
            or tuple(
                (record.evidence_id, record.content_hash)
                for record in self.source_evidence
            )
            != tuple(
                (member.evidence_id, member.evidence_content_hash)
                for member in snapshot.members
            )
            or any(record.event.scope != scope for record in self.source_evidence)
            or self.result_release.manifest.scope != scope
            or application.result_release_id != self.result_release.release_id
            or application.result_release_content_sha256
            != self.result_release.content_hash
            or application.result_revision_ids
            != self.result_release.manifest.revision_ids
            or tuple(item.revision_id for item in self.result_revisions)
            != self.result_release.manifest.revision_ids
            or any(item.proposal.scope != scope for item in self.result_revisions)
            or len({item.memory_id for item in self.result_revisions})
            != len(self.result_revisions)
        ):
            raise ValueError("replay step commitments do not agree")
        member_by_id = {
            member.evidence_id: member for member in snapshot.members
        }
        if any(
            member_by_id.get(grounding.evidence_id) != grounding
            for update in application.applied_updates
            for grounding in update.grounding
        ):
            raise ValueError("application grounding is absent from its source snapshot")
        for update in application.applied_updates:
            if update.release_position >= len(self.result_revisions):
                raise ValueError("applied update result position is out of range")
            revision = self.result_revisions[update.release_position]
            if (
                revision.revision_id != update.revision_id
                or revision.content_hash != update.revision_content_sha256
                or revision.memory_id != update.memory_id
                or revision.generation != update.generation
                or revision.proposal.operation is not update.operation
                or revision.proposal.parent_revision_id != update.parent_revision_id
                or revision.proposal.candidate_id != update.candidate_id
            ):
                raise ValueError("applied update disagrees with its result revision")


@dataclass(frozen=True, slots=True)
class MemoryApplicationReplayViewV1:
    """A root-to-target linear lineage loaded from one consistent read view."""

    root: MemoryApplicationRootV1
    root_release: MemoryRelease
    root_revisions: tuple[MemoryRevision, ...]
    steps: tuple[MemoryApplicationReplayStepV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.root) is not MemoryApplicationRootV1
            or type(self.root_release) is not MemoryRelease
            or type(self.root_revisions) is not tuple
            or any(type(item) is not MemoryRevision for item in self.root_revisions)
            or type(self.steps) is not tuple
            or any(type(item) is not MemoryApplicationReplayStepV1 for item in self.steps)
        ):
            raise TypeError("replay view values must use exact Memory Service types")
        scope = self.root.scope
        if (
            self.root_release.manifest.scope != scope
            or self.root.release_id != self.root_release.release_id
            or self.root.release_content_sha256 != self.root_release.content_hash
            or tuple(item.revision_id for item in self.root_revisions)
            != self.root_release.manifest.revision_ids
            or any(item.proposal.scope != scope for item in self.root_revisions)
            or len({item.memory_id for item in self.root_revisions})
            != len(self.root_revisions)
        ):
            raise ValueError("replay root commitments do not agree")
        current_release = self.root_release
        previous_order: int | None = None
        application_ids: set[str] = set()
        release_ids = {current_release.release_id}
        for step in self.steps:
            application = step.application
            if (
                application.proposal.scope != scope
                or application.proposal.source_base_release_id
                != current_release.release_id
                or application.base_release_content_sha256
                != current_release.content_hash
                or application.application_id in application_ids
                or step.result_release.release_id in release_ids
                or (
                    previous_order is not None
                    and application.application_order <= previous_order
                )
            ):
                raise ValueError("replay steps do not form one forward lineage")
            application_ids.add(application.application_id)
            release_ids.add(step.result_release.release_id)
            previous_order = application.application_order
            current_release = step.result_release

    @property
    def target_release(self) -> MemoryRelease:
        """Return the requested lineage target release."""

        if self.steps:
            return self.steps[-1].result_release
        return self.root_release

    @property
    def target_revisions(self) -> tuple[MemoryRevision, ...]:
        """Return the target release's exact ordered revisions."""

        if self.steps:
            return self.steps[-1].result_revisions
        return self.root_revisions
