# SPDX-License-Identifier: Apache-2.0

"""Public contracts for immutable evidence, history, releases, and applications."""

from areal.v2.memory_service.application_replay import (
    MemoryApplicationReplayStepV1,
    MemoryApplicationReplayViewV1,
)
from areal.v2.memory_service.application_store import MemoryApplicationStore
from areal.v2.memory_service.application_types import (
    AppliedMemoryUpdateV1,
    MemoryApplicationProposal,
    MemoryApplicationRootV1,
    MemoryApplicationUpdateProposal,
    MemoryApplicationV1,
)
from areal.v2.memory_service.errors import (
    CandidateConflictError,
    CandidateNotFoundError,
    EvidenceConflictError,
    EvidenceNotFoundError,
    EvidenceSnapshotConflictError,
    EvidenceSnapshotNotFoundError,
    MemoryApplicationConflictError,
    MemoryApplicationNotFoundError,
    MemoryApplicationRootConflictError,
    MemoryApplicationRootNotFoundError,
    MemoryApplicationStaleSnapshotError,
    MemoryServiceError,
    ReleaseConflictError,
    ReleaseNotFoundError,
    RevisionConflictError,
    RevisionNotFoundError,
)
from areal.v2.memory_service.history_store import (
    InMemoryMemoryHistoryStore,
    MemoryHistoryStore,
)
from areal.v2.memory_service.history_types import (
    CandidateProposal,
    MemoryCandidate,
    MemoryRevision,
    RevisionOperation,
    RevisionProposal,
)
from areal.v2.memory_service.release_store import (
    InMemoryMemoryReleaseStore,
    MemoryReleaseStore,
)
from areal.v2.memory_service.release_types import MemoryRelease, ReleaseManifest
from areal.v2.memory_service.snapshot_types import (
    EVIDENCE_SNAPSHOT_ORDERING_POLICY,
    EvidenceSnapshot,
    EvidenceSnapshotMember,
    EvidenceSnapshotSpec,
)
from areal.v2.memory_service.store import (
    EvidenceSnapshotStore,
    EvidenceStore,
    InMemoryEvidenceStore,
)
from areal.v2.memory_service.types import (
    EvidenceEvent,
    EvidenceKind,
    EvidenceRecord,
    MemoryScope,
)

__all__ = [
    "AppliedMemoryUpdateV1",
    "CandidateConflictError",
    "CandidateNotFoundError",
    "CandidateProposal",
    "EVIDENCE_SNAPSHOT_ORDERING_POLICY",
    "EvidenceConflictError",
    "EvidenceEvent",
    "EvidenceKind",
    "EvidenceNotFoundError",
    "EvidenceRecord",
    "EvidenceSnapshot",
    "EvidenceSnapshotConflictError",
    "EvidenceSnapshotMember",
    "EvidenceSnapshotNotFoundError",
    "EvidenceSnapshotSpec",
    "EvidenceSnapshotStore",
    "EvidenceStore",
    "InMemoryEvidenceStore",
    "InMemoryMemoryHistoryStore",
    "InMemoryMemoryReleaseStore",
    "MemoryCandidate",
    "MemoryApplicationConflictError",
    "MemoryApplicationNotFoundError",
    "MemoryApplicationProposal",
    "MemoryApplicationReplayStepV1",
    "MemoryApplicationReplayViewV1",
    "MemoryApplicationRootConflictError",
    "MemoryApplicationRootNotFoundError",
    "MemoryApplicationRootV1",
    "MemoryApplicationStaleSnapshotError",
    "MemoryApplicationStore",
    "MemoryApplicationUpdateProposal",
    "MemoryApplicationV1",
    "MemoryHistoryStore",
    "MemoryRelease",
    "MemoryReleaseStore",
    "MemoryRevision",
    "MemoryScope",
    "MemoryServiceError",
    "ReleaseConflictError",
    "ReleaseManifest",
    "ReleaseNotFoundError",
    "RevisionConflictError",
    "RevisionNotFoundError",
    "RevisionOperation",
    "RevisionProposal",
]
