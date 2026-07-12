# SPDX-License-Identifier: Apache-2.0

"""Trusted runtime renderer/consumer adapters for the frozen codebook task.

These adapters let the production-shaped query/exposure contracts execute the
existing deterministic helpfulness task without changing its renderer or
consumer semantics.  They are evaluation adapters, not a general Agent
Service transport and not evidence that Memory improves a real model.
"""

from __future__ import annotations

import hashlib

from examples.memory_service import scoped_codebook_eval as harness

from areal.v2.memory_service import (
    MemoryConsumerCallV1,
    MemoryConsumerKind,
    MemoryQueryResultV1,
    MemoryRenderedRevisionRangeV1,
    MemoryRenderOutputV1,
    MemoryRetrievalOutputV1,
)

RELEASE_ORDER_RETRIEVER_ID_V1 = "release-manifest-order"
RELEASE_ORDER_RETRIEVER_VERSION_SHA256_V1 = hashlib.sha256(
    b"release-order-retrieval-v1"
).hexdigest()
CODEBOOK_RENDERER_ID_V1 = "memory-codebook/v1"
CODEBOOK_RENDERER_VERSION_SHA256_V1 = hashlib.sha256(
    b"areal-memory-codebook-runtime-renderer-v1"
).hexdigest()
SCRIPTED_CONSUMER_ID_V1 = "scripted-last-occurrence/v1"
SCRIPTED_CONSUMER_VERSION_SHA256_V1 = hashlib.sha256(
    b"areal-memory-codebook-runtime-consumer-v1"
).hexdigest()


def _history_sha256(history: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256(b"areal-memory-runtime-history-v1\0")
    digest.update(len(history).to_bytes(8, "big"))
    for item in history:
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return digest.hexdigest()


class ReleaseManifestRetrieverV1:
    """Return every eligible revision in the pinned manifest order."""

    retrieval_policy_id = RELEASE_ORDER_RETRIEVER_ID_V1
    retrieval_policy_version_sha256 = RELEASE_ORDER_RETRIEVER_VERSION_SHA256_V1

    def retrieve(self, *, attempt, query, eligible_items):
        del attempt, query
        revision_ids = tuple(item.revision.revision_id for item in eligible_items)
        return MemoryRetrievalOutputV1(
            retrieved_revision_ids=revision_ids,
            returned_revision_ids=revision_ids,
        )


class CodebookReleaseRendererV1:
    """Render store-authentic query items with the frozen codebook grammar."""

    renderer_id = CODEBOOK_RENDERER_ID_V1
    renderer_version_sha256 = CODEBOOK_RENDERER_VERSION_SHA256_V1

    def render(self, query_result: MemoryQueryResultV1) -> MemoryRenderOutputV1:
        if type(query_result) is not MemoryQueryResultV1:
            raise TypeError("query_result must be a MemoryQueryResultV1")
        entries = []
        for item in query_result.returned_items:
            parsed = harness.parse_fact(item.content)
            entries.append(
                harness.ResolvedEntry(
                    slot=item.release_position,
                    key=parsed.key,
                    value=parsed.value,
                    source_kind="release",
                    revision_id=item.revision.revision_id,
                    candidate_id=item.candidate_id,
                    evidence_ids=tuple(
                        evidence.evidence_id for evidence in item.evidence
                    ),
                )
            )
        rendered = harness.render_context(tuple(entries))
        return MemoryRenderOutputV1(
            rendered_context=rendered.bytes,
            rendered_ranges=tuple(
                MemoryRenderedRevisionRangeV1(
                    revision_id=receipt.revision_id,
                    rendered_start=receipt.rendered_start,
                    rendered_end=receipt.rendered_end,
                )
                for receipt in rendered.entry_receipts
                if receipt.revision_id is not None
            ),
        )


class ScriptedCodebookConsumerV1:
    """Actually invoke the frozen scripted consumer at the receipt boundary."""

    consumer_kind = MemoryConsumerKind.CONTEXT
    consumer_id = SCRIPTED_CONSUMER_ID_V1
    consumer_version_sha256 = SCRIPTED_CONSUMER_VERSION_SHA256_V1

    def submit(
        self,
        *,
        delivery,
        rendered_context: bytes,
        query: bytes,
        history: tuple[bytes, ...],
        call_id: str,
    ) -> MemoryConsumerCallV1:
        result = harness.consume_scripted(
            query,
            rendered_context,
            history=history,
        )
        return MemoryConsumerCallV1(
            delivery_id=delivery.delivery_id,
            delivery_content_sha256=delivery.content_hash,
            call_id=call_id,
            submitted_prompt=rendered_context,
            context_start=0,
            context_end=len(rendered_context),
            observed_query_sha256=hashlib.sha256(query).hexdigest(),
            observed_history_sha256=_history_sha256(history),
            observed_history_length=len(history),
            input_token_ids=None,
            output=result.response,
        )


__all__ = [
    "CODEBOOK_RENDERER_ID_V1",
    "CODEBOOK_RENDERER_VERSION_SHA256_V1",
    "RELEASE_ORDER_RETRIEVER_ID_V1",
    "RELEASE_ORDER_RETRIEVER_VERSION_SHA256_V1",
    "SCRIPTED_CONSUMER_ID_V1",
    "SCRIPTED_CONSUMER_VERSION_SHA256_V1",
    "CodebookReleaseRendererV1",
    "ReleaseManifestRetrieverV1",
    "ScriptedCodebookConsumerV1",
]
