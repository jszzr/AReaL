# SPDX-License-Identifier: Apache-2.0

"""Storage contract for causally sealed, atomic Memory applications."""

from __future__ import annotations

from typing import Protocol

from areal.v2.memory_service.application_types import (
    MemoryApplicationProposal,
    MemoryApplicationRootV1,
    MemoryApplicationV1,
)
from areal.v2.memory_service.types import MemoryScope


class MemoryApplicationStore(Protocol):
    """Optional extension for releases produced by a sealed policy decision."""

    def register_memory_application_root(
        self,
        scope: MemoryScope,
        release_id: str,
    ) -> MemoryApplicationRootV1:
        """Commit the one explicit non-application release trusted by a scope."""

        ...

    def get_memory_application_root(
        self,
        scope: MemoryScope,
    ) -> MemoryApplicationRootV1:
        """Load the exact immutable application root for one scope."""

        ...

    def commit_memory_application(
        self,
        proposal: MemoryApplicationProposal,
    ) -> MemoryApplicationV1:
        """Atomically publish updates, their release, and provenance ledger."""

        ...

    def get_memory_application(
        self,
        scope: MemoryScope,
        application_id: str,
    ) -> MemoryApplicationV1:
        """Load one committed application by exact scope and content ID."""

        ...

    def get_memory_application_for_revision(
        self,
        scope: MemoryScope,
        revision_id: str,
    ) -> MemoryApplicationV1:
        """Resolve the unique application that created a revision."""

        ...

    def get_memory_application_for_release(
        self,
        scope: MemoryScope,
        release_id: str,
    ) -> MemoryApplicationV1:
        """Resolve the unique application that published a release."""

        ...
