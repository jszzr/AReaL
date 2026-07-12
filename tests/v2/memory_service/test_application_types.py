# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from areal.v2.memory_service.application_types import (
    AppliedMemoryUpdateV1,
    MemoryApplicationProposal,
    MemoryApplicationRootV1,
    MemoryApplicationUpdateProposal,
    MemoryApplicationV1,
)
from areal.v2.memory_service.history_types import RevisionOperation
from areal.v2.memory_service.snapshot_types import EvidenceSnapshotMember
from areal.v2.memory_service.types import MemoryScope

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
CREATED_AT = datetime(2026, 7, 12, 3, 4, 5, 6000, tzinfo=UTC)


def make_scope() -> MemoryScope:
    return MemoryScope("tenant", "agent-memory", "agent-1")


def make_member(
    evidence_id: str = "ev_1",
    evidence_content_hash: str = HASH_A,
    ingest_order: int = 7,
) -> EvidenceSnapshotMember:
    return EvidenceSnapshotMember(
        evidence_id=evidence_id,
        evidence_content_hash=evidence_content_hash,
        ingest_order=ingest_order,
    )


def make_update(**overrides: object) -> MemoryApplicationUpdateProposal:
    values: dict[str, object] = {
        "content": "preferred_timezone=Asia/Shanghai",
        "evidence_ids": ("ev_1",),
        "operation": RevisionOperation.ADD,
        "parent_revision_id": None,
    }
    values.update(overrides)
    return MemoryApplicationUpdateProposal(**values)  # type: ignore[arg-type]


def make_proposal(**overrides: object) -> MemoryApplicationProposal:
    values: dict[str, object] = {
        "scope": make_scope(),
        "source_snapshot_id": "snap_1",
        "source_base_release_id": "rel_base",
        "projector_id": "provenance-projector-v1",
        "projector_version_sha256": HASH_A,
        "policy_id": "verified-chain-consensus-v1",
        "policy_version_sha256": HASH_B,
        "policy_input_sha256": HASH_C,
        "decision_sha256": HASH_D,
        "policy_context": '{"minimum_support":1,"mode":"consensus"}',
        "updates": (make_update(),),
        "idempotency_key": "apply-run-1",
    }
    values.update(overrides)
    return MemoryApplicationProposal(**values)  # type: ignore[arg-type]


def make_applied(**overrides: object) -> AppliedMemoryUpdateV1:
    values: dict[str, object] = {
        "ordinal": 0,
        "release_position": 1,
        "operation": RevisionOperation.ADD,
        "grounding": (make_member(),),
        "candidate_id": "cand_1",
        "candidate_content_sha256": HASH_B,
        "revision_id": "rev_new",
        "revision_content_sha256": HASH_C,
        "memory_id": "mem_1",
        "generation": 0,
        "parent_revision_id": None,
    }
    values.update(overrides)
    return AppliedMemoryUpdateV1(**values)  # type: ignore[arg-type]


def make_application(**overrides: object) -> MemoryApplicationV1:
    values: dict[str, object] = {
        "proposal": make_proposal(),
        "source_snapshot_content_sha256": HASH_E,
        "source_evidence_high_watermark": 9,
        "base_release_content_sha256": HASH_A,
        "result_release_id": "rel_result",
        "result_release_content_sha256": HASH_B,
        "result_revision_ids": ("rev_existing", "rev_new"),
        "applied_updates": (make_applied(),),
        "application_order": 4,
        "created_at": CREATED_AT,
    }
    values.update(overrides)
    return MemoryApplicationV1.create(**values)  # type: ignore[arg-type]


def test_update_proposal_canonical_bytes_are_stable_and_complete() -> None:
    update = make_update()

    assert json.loads(update.canonical_bytes()) == {
        "schema_version": 1,
        "content": "preferred_timezone=Asia/Shanghai",
        "evidence_ids": ["ev_1"],
        "operation": "add",
        "parent_revision_id": None,
    }
    assert update.canonical_bytes() == make_update().canonical_bytes()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"content": "  "}, ValueError),
        ({"evidence_ids": []}, TypeError),
        ({"evidence_ids": ()}, ValueError),
        ({"evidence_ids": ("ev_1", "ev_1")}, ValueError),
        ({"operation": "add"}, TypeError),
        ({"operation": RevisionOperation.REFINE}, ValueError),
        ({"operation": RevisionOperation.CONTRADICT}, ValueError),
        ({"parent_revision_id": "rev_parent"}, ValueError),
        (
            {
                "operation": RevisionOperation.SUPERSEDE,
                "parent_revision_id": None,
            },
            ValueError,
        ),
    ],
)
def test_update_proposal_rejects_invalid_or_ambiguous_values(
    overrides: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        make_update(**overrides)


def test_supersede_update_is_supported() -> None:
    update = make_update(
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id="rev_parent",
    )

    assert json.loads(update.canonical_bytes())["operation"] == "supersede"
    assert json.loads(update.canonical_bytes())["parent_revision_id"] == "rev_parent"


def test_application_proposal_commits_every_policy_identity_and_input() -> None:
    proposal = make_proposal()
    value = json.loads(proposal.canonical_bytes())

    assert value["schema_version"] == 1
    assert value["scope"] == {
        "namespace": "agent-memory",
        "subject_id": "agent-1",
        "tenant_id": "tenant",
    }
    assert value["source_snapshot_id"] == "snap_1"
    assert value["source_base_release_id"] == "rel_base"
    assert value["projector_version_sha256"] == HASH_A
    assert value["policy_version_sha256"] == HASH_B
    assert value["policy_input_sha256"] == HASH_C
    assert value["decision_sha256"] == HASH_D
    assert value["policy_context"] == ('{"minimum_support":1,"mode":"consensus"}')
    assert value["updates"][0]["evidence_ids"] == ["ev_1"]


@pytest.mark.parametrize(
    "policy_context",
    [
        "",
        ' {"a":1}',
        '{"b":2,"a":1}',
        '{"a": 1}',
        '{"a":1,"a":2}',
        '{"a":NaN}',
        "not-json",
    ],
)
def test_application_proposal_requires_exact_canonical_policy_context(
    policy_context: str,
) -> None:
    with pytest.raises(ValueError):
        make_proposal(policy_context=policy_context)


@pytest.mark.parametrize(
    "field_name",
    [
        "projector_version_sha256",
        "policy_version_sha256",
        "policy_input_sha256",
        "decision_sha256",
    ],
)
def test_application_proposal_requires_full_lowercase_sha256(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        make_proposal(**{field_name: "A" * 64})


def test_application_proposal_rejects_duplicate_updates() -> None:
    update = make_update()
    with pytest.raises(ValueError, match="updates must not contain duplicates"):
        make_proposal(updates=(update, update))


def test_application_proposal_rejects_two_supersedes_of_one_parent() -> None:
    first = make_update(
        content="fact=one",
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id="rev_parent",
    )
    second = make_update(
        content="fact=two",
        evidence_ids=("ev_2",),
        operation=RevisionOperation.SUPERSEDE,
        parent_revision_id="rev_parent",
    )

    with pytest.raises(ValueError, match="same parent"):
        make_proposal(updates=(first, second))


def test_applied_update_commits_full_grounding_and_object_hashes() -> None:
    applied = make_applied()
    value = json.loads(applied.canonical_bytes())

    assert value == {
        "candidate_content_sha256": HASH_B,
        "candidate_id": "cand_1",
        "generation": 0,
        "grounding": [
            {
                "evidence_content_hash": HASH_A,
                "evidence_id": "ev_1",
                "ingest_order": 7,
            }
        ],
        "memory_id": "mem_1",
        "operation": "add",
        "ordinal": 0,
        "parent_revision_id": None,
        "release_position": 1,
        "revision_content_sha256": HASH_C,
        "revision_id": "rev_new",
        "schema_version": 1,
    }


def test_applied_update_rejects_invalid_generation_for_operation() -> None:
    with pytest.raises(ValueError, match="ADD generation"):
        make_applied(generation=1)
    with pytest.raises(ValueError, match="SUPERSEDE generation"):
        make_applied(
            operation=RevisionOperation.SUPERSEDE,
            parent_revision_id="rev_parent",
            generation=0,
        )


def test_applied_update_requires_unique_full_grounding_members() -> None:
    with pytest.raises(ValueError, match="duplicate evidence IDs"):
        make_applied(grounding=(make_member(), make_member(ingest_order=8)))
    with pytest.raises(ValueError, match="duplicate ingest orders"):
        make_applied(
            grounding=(
                make_member(),
                make_member("ev_2", HASH_B, ingest_order=7),
            )
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        make_applied(grounding=(make_member(evidence_content_hash="short"),))


def test_create_content_addresses_the_complete_atomic_application() -> None:
    application = make_application()
    canonical = application.canonical_bytes()
    expected_hash = hashlib.sha256(canonical).hexdigest()
    value = json.loads(canonical)

    assert application.content_hash == expected_hash
    assert application.application_id == f"mapp_{expected_hash[:24]}"
    assert value["application_order"] == 4
    assert value["proposal"]["source_snapshot_id"] == "snap_1"
    assert value["source_snapshot_content_sha256"] == HASH_E
    assert value["source_evidence_high_watermark"] == 9
    assert value["base_release_content_sha256"] == HASH_A
    assert value["result_release_id"] == "rel_result"
    assert value["result_release_content_sha256"] == HASH_B
    assert value["result_revision_ids"] == ["rev_existing", "rev_new"]
    assert value["applied_updates"][0]["revision_id"] == "rev_new"


def test_application_order_is_part_of_the_content_address() -> None:
    first = make_application(application_order=4)
    second = make_application(application_order=5)

    assert first.canonical_bytes() != second.canonical_bytes()
    assert first.content_hash != second.content_hash
    assert first.application_id != second.application_id


def test_created_at_is_normalized_but_is_storage_metadata() -> None:
    plus_eight = timezone(timedelta(hours=8))
    first = make_application(created_at=CREATED_AT)
    second = make_application(created_at=CREATED_AT.astimezone(plus_eight))

    assert second.created_at == CREATED_AT
    assert second.created_at.tzinfo is UTC
    assert first.canonical_bytes() == second.canonical_bytes()


def test_stored_fields_reconstruct_and_reverify_the_application() -> None:
    original = make_application()
    reconstructed = MemoryApplicationV1(
        proposal=original.proposal,
        source_snapshot_content_sha256=original.source_snapshot_content_sha256,
        source_evidence_high_watermark=original.source_evidence_high_watermark,
        base_release_content_sha256=original.base_release_content_sha256,
        result_release_id=original.result_release_id,
        result_release_content_sha256=original.result_release_content_sha256,
        result_revision_ids=original.result_revision_ids,
        applied_updates=original.applied_updates,
        application_order=original.application_order,
        application_id=original.application_id,
        content_hash=original.content_hash,
        created_at=original.created_at,
    )

    assert reconstructed == original
    assert reconstructed.canonical_bytes() == original.canonical_bytes()


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("application_id", "mapp_wrong", "application_id disagrees"),
        ("content_hash", HASH_D, "content_hash disagrees"),
    ],
)
def test_reconstruction_rejects_identity_drift(
    field_name: str,
    field_value: str,
    message: str,
) -> None:
    original = make_application()
    values = {
        "proposal": original.proposal,
        "source_snapshot_content_sha256": original.source_snapshot_content_sha256,
        "source_evidence_high_watermark": original.source_evidence_high_watermark,
        "base_release_content_sha256": original.base_release_content_sha256,
        "result_release_id": original.result_release_id,
        "result_release_content_sha256": original.result_release_content_sha256,
        "result_revision_ids": original.result_revision_ids,
        "applied_updates": original.applied_updates,
        "application_order": original.application_order,
        "application_id": original.application_id,
        "content_hash": original.content_hash,
        "created_at": original.created_at,
    }
    values[field_name] = field_value

    with pytest.raises(ValueError, match=message):
        MemoryApplicationV1(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("proposal", "applied", "message"),
    [
        (
            make_proposal(),
            make_applied(
                operation=RevisionOperation.SUPERSEDE,
                parent_revision_id="rev_parent",
                generation=1,
            ),
            "operation drifted",
        ),
        (
            make_proposal(
                updates=(
                    make_update(
                        operation=RevisionOperation.SUPERSEDE,
                        parent_revision_id="rev_parent",
                    ),
                )
            ),
            make_applied(
                operation=RevisionOperation.SUPERSEDE,
                parent_revision_id="rev_other",
                generation=1,
            ),
            "parent drifted",
        ),
        (
            make_proposal(),
            make_applied(grounding=(make_member("ev_other"),)),
            "grounding drifted",
        ),
    ],
)
def test_create_rejects_applied_update_drift(
    proposal: MemoryApplicationProposal,
    applied: AppliedMemoryUpdateV1,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_application(proposal=proposal, applied_updates=(applied,))


def test_create_rejects_release_membership_drift() -> None:
    with pytest.raises(ValueError, match="release membership drifted"):
        make_application(result_revision_ids=("rev_existing", "rev_other"))
    with pytest.raises(ValueError, match="address result_revision_ids"):
        make_application(applied_updates=(make_applied(release_position=2),))


def test_create_rejects_grounding_beyond_snapshot_high_watermark() -> None:
    with pytest.raises(ValueError, match="exceeds source_evidence_high_watermark"):
        make_application(source_evidence_high_watermark=6)


def test_create_rejects_duplicate_result_or_applied_object_ids() -> None:
    with pytest.raises(ValueError, match="result_revision_ids.*duplicates"):
        make_application(result_revision_ids=("rev_new", "rev_new"))

    update_2 = make_update(content="language=zh", evidence_ids=("ev_2",))
    proposal = make_proposal(updates=(make_update(), update_2))
    applied_2 = make_applied(
        ordinal=1,
        release_position=2,
        grounding=(make_member("ev_2", HASH_D, 8),),
        revision_id="rev_new_2",
        revision_content_sha256=HASH_D,
        memory_id="mem_2",
    )
    with pytest.raises(ValueError, match="candidate IDs"):
        make_application(
            proposal=proposal,
            result_revision_ids=("rev_existing", "rev_new", "rev_new_2"),
            applied_updates=(make_applied(), applied_2),
        )


def test_canonical_bytes_fail_closed_after_nested_or_identity_mutation() -> None:
    application = make_application()
    object.__setattr__(application.applied_updates[0], "revision_id", "rev_drift")
    with pytest.raises(ValueError, match="release membership drifted"):
        application.canonical_bytes()

    application = make_application()
    object.__setattr__(application, "content_hash", HASH_D)
    with pytest.raises(ValueError, match="content_hash disagrees"):
        application.canonical_bytes()


def test_application_requires_exact_runtime_types() -> None:
    with pytest.raises(TypeError, match="application_order must be an integer"):
        make_application(application_order=True)
    with pytest.raises(TypeError, match="result_revision_ids must be a tuple"):
        make_application(result_revision_ids=["rev_existing", "rev_new"])
    with pytest.raises(TypeError, match="applied_updates must be a tuple"):
        make_application(applied_updates=[make_applied()])


def test_root_create_content_addresses_the_release_anchor() -> None:
    root = MemoryApplicationRootV1.create(
        scope=make_scope(),
        release_id="rel_genesis",
        release_content_sha256=HASH_A,
        created_at=CREATED_AT,
    )
    canonical = root.canonical_bytes()
    expected_hash = hashlib.sha256(canonical).hexdigest()

    assert root.root_id == f"mroot_{expected_hash[:24]}"
    assert root.content_hash == expected_hash
    assert json.loads(canonical) == {
        "release_content_sha256": HASH_A,
        "release_id": "rel_genesis",
        "schema_version": 1,
        "scope": {
            "namespace": "agent-memory",
            "subject_id": "agent-1",
            "tenant_id": "tenant",
        },
    }


def test_root_reconstructs_and_rejects_hash_or_id_drift() -> None:
    root = MemoryApplicationRootV1.create(
        scope=make_scope(),
        release_id="rel_genesis",
        release_content_sha256=HASH_A,
        created_at=CREATED_AT,
    )
    assert replace(root) == root

    with pytest.raises(ValueError, match="root_id disagrees"):
        replace(root, root_id="mroot_wrong")
    with pytest.raises(ValueError, match="content_hash disagrees"):
        replace(root, content_hash=HASH_D)


def test_root_canonical_bytes_fail_closed_after_mutation() -> None:
    root = MemoryApplicationRootV1.create(
        scope=make_scope(),
        release_id="rel_genesis",
        release_content_sha256=HASH_A,
        created_at=CREATED_AT,
    )
    object.__setattr__(root, "release_id", "rel_drift")

    with pytest.raises(ValueError, match="content_hash disagrees"):
        root.canonical_bytes()
