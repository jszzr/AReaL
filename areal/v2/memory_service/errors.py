# SPDX-License-Identifier: Apache-2.0

"""Errors raised by the Memory Service."""

from __future__ import annotations


class MemoryServiceError(Exception):
    """Base class for Memory Service failures."""


class EvidenceNotFoundError(MemoryServiceError):
    """Raised when requested evidence is unavailable in the requested scope."""


class EvidenceConflictError(MemoryServiceError):
    """Raised when evidence conflicts with an existing immutable record."""


class CandidateNotFoundError(MemoryServiceError):
    """Raised when a candidate is unavailable in the requested scope."""


class CandidateConflictError(MemoryServiceError):
    """Raised when a candidate write conflicts with immutable history."""


class RevisionNotFoundError(MemoryServiceError):
    """Raised when a revision is unavailable in the requested scope."""


class RevisionConflictError(MemoryServiceError):
    """Raised when a revision write conflicts with immutable history."""


class ReleaseNotFoundError(MemoryServiceError):
    """Raised when a release is unavailable in the requested scope."""


class ReleaseConflictError(MemoryServiceError):
    """Raised when a release write conflicts with immutable history."""


class MemoryPersistenceError(MemoryServiceError):
    """Base class for durable Memory Service storage failures."""


class MemoryPersistenceBusyError(MemoryPersistenceError):
    """Raised when SQLite cannot acquire the required local database lock."""


class MemoryPersistenceSchemaError(MemoryPersistenceError):
    """Raised when a database does not have the exact supported schema."""


class MemoryPersistenceCorruptionError(MemoryPersistenceError):
    """Raised when durable data violates its stored integrity contract."""


class MemoryQueryNotFoundError(MemoryServiceError):
    """Raised when a runtime query attempt or result cannot be resolved."""


class MemoryQueryConflictError(MemoryServiceError):
    """Raised when an immutable runtime query stage conflicts."""


class MemoryDeliveryNotFoundError(MemoryServiceError):
    """Raised when a rendered Memory delivery cannot be resolved."""


class MemoryDeliveryConflictError(MemoryServiceError):
    """Raised when an immutable rendered Memory delivery conflicts."""


class MemoryConsumerAckNotFoundError(MemoryServiceError):
    """Raised when a consumer-boundary acknowledgement cannot be resolved."""


class MemoryConsumerAckConflictError(MemoryServiceError):
    """Raised when a consumer acknowledgement conflicts or is replayed."""


class MemoryExposureNotFoundError(MemoryServiceError):
    """Raised when an actual Memory exposure cannot be resolved."""


class MemoryExposureConflictError(MemoryServiceError):
    """Raised when an actual Memory exposure chain conflicts."""


class MemoryBoundaryMismatchError(MemoryServiceError):
    """Raised when submitted consumer bytes do not match a pending delivery."""
