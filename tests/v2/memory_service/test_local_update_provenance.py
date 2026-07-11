# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json

import pytest

from examples.memory_service import local_update_provenance as provenance

TOOL_VERSION = "a" * 64
AGENT_VERSION = "b" * 64
EVALUATOR_VERSION = "c" * 64
RECEIPT = "d" * 64


def _receipt() -> provenance.OpaqueReceiptCommitmentV1:
    return provenance.OpaqueReceiptCommitmentV1(
        kind="local_test_receipt",
        schema_version=1,
        byte_count=32,
        sha256=RECEIPT,
    )


def _link(
    relation: provenance.ProvenanceRelationV1,
    *,
    digit: str = "1",
) -> provenance.ProvenanceLinkV1:
    return provenance.ProvenanceLinkV1(
        relation=relation,
        target_evidence_id=f"evd_{digit * 24}",
        target_evidence_content_sha256=digit * 64,
    )


def _claim() -> provenance.ProvenancePayloadV1:
    return provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="trajectory-1",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.USER,
            producer_id="subject-1",
            version_sha256=None,
        ),
        links=(),
        body=provenance.FactClaimBodyV1(
            fact_namespace="project_registry",
            key="project-alpha",
            value="NEW42",
        ),
        receipt=None,
    )


def _call() -> provenance.ProvenancePayloadV1:
    request_content_sha256 = provenance.tool_request_content_sha256_v1(
        tool_name="code_registry.lookup",
        tool_version_sha256=TOOL_VERSION,
        fact_namespace="project_registry",
        key="project-alpha",
    )
    return provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="trajectory-1",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.AGENT,
            producer_id="memory-agent",
            version_sha256=AGENT_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.TRIGGERED_BY),),
        body=provenance.ToolCallBodyV1(
            call_id="call-1",
            tool_name="code_registry.lookup",
            tool_version_sha256=TOOL_VERSION,
            fact_namespace="project_registry",
            key="project-alpha",
            request_content_sha256=request_content_sha256,
        ),
        receipt=None,
    )


def _result(
    *,
    status: provenance.ToolResultStatusV1 = provenance.ToolResultStatusV1.OK,
    value: str | None = "NEW42",
    error_code: str | None = None,
) -> provenance.ProvenancePayloadV1:
    if status is provenance.ToolResultStatusV1.ERROR and error_code is None:
        error_code = "lookup_error"
    result_content_sha256 = provenance.tool_result_content_sha256_v1(
        call_id="call-1",
        tool_name="code_registry.lookup",
        tool_version_sha256=TOOL_VERSION,
        status=status,
        fact_namespace="project_registry",
        key="project-alpha",
        value=value,
        error_code=error_code,
    )
    return provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="trajectory-1",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.TOOL,
            producer_id="code_registry.lookup",
            version_sha256=TOOL_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.RESULT_OF, digit="2"),),
        body=provenance.ToolResultBodyV1(
            call_id="call-1",
            tool_name="code_registry.lookup",
            tool_version_sha256=TOOL_VERSION,
            status=status,
            fact_namespace="project_registry",
            key="project-alpha",
            value=value,
            error_code=error_code,
            result_content_sha256=result_content_sha256,
        ),
        receipt=_receipt(),
    )


def _outcome(
    verdict: provenance.VerificationVerdictV1 = provenance.VerificationVerdictV1.PASS,
) -> provenance.ProvenancePayloadV1:
    return provenance.ProvenancePayloadV1(
        schema_version=1,
        trajectory_id="trajectory-1",
        producer=provenance.ProvenanceProducerV1(
            kind=provenance.ProducerKindV1.EVALUATOR,
            producer_id="code_registry.verifier",
            version_sha256=EVALUATOR_VERSION,
        ),
        links=(_link(provenance.ProvenanceRelationV1.EVALUATES, digit="3"),),
        body=provenance.VerificationOutcomeBodyV1(
            outcome_type="claim_verification",
            verdict=verdict,
            evaluator_id="code_registry.verifier",
            evaluator_version_sha256=EVALUATOR_VERSION,
        ),
        receipt=_receipt(),
    )


def test_fact_claim_has_locked_canonical_wire_and_hash() -> None:
    value = _claim()
    wire = provenance.provenance_payload_wire_v1(value)
    assert wire == (
        '{"body":{"fact_namespace":"project_registry","key":"project-alpha",'
        '"value":"NEW42"},"links":[],"producer":{"kind":"user",'
        '"producer_id":"subject-1","version_sha256":null},'
        '"receipt":null,"schema_version":1,'
        '"trajectory_id":"trajectory-1","type":"fact_claim"}'
    )
    assert provenance.provenance_payload_sha256_v1(value) == (
        "698b7bfcc410f80bf918bd1b28be468c4dbd7388c6ddbaeddbf385796c33be33"
    )
    assert provenance.parse_provenance_payload_v1(wire) == value


def test_tool_call_and_result_payloads_have_locked_golden_vectors() -> None:
    call_wire = (
        '{"body":{"call_id":"call-1","fact_namespace":"project_registry",'
        '"key":"project-alpha","request_content_sha256":'
        '"d4938686ea6da366b2ff1b65eb699fc955d17c2565440359f71621285c8338d2",'
        '"tool_name":"code_registry.lookup","tool_version_sha256":'
        f'"{TOOL_VERSION}"}},"links":[{{"relation":"triggered_by",'
        '"target_evidence_content_sha256":'
        f'"{"1" * 64}","target_evidence_id":"evd_{"1" * 24}"}}],'
        f'"producer":{{"kind":"agent","producer_id":"memory-agent",'
        f'"version_sha256":"{AGENT_VERSION}"}},"receipt":null,'
        '"schema_version":1,"trajectory_id":"trajectory-1","type":"tool_call"}'
    )
    ok_prefix = (
        '{"body":{"call_id":"call-1","error_code":null,'
        '"fact_namespace":"project_registry","key":"project-alpha",'
        '"result_content_sha256":'
        '"e9e1147029a6cee781cacfb1015f720e05b7d256eab43b0939a638ae0f0477f7",'
        '"status":"ok","tool_name":"code_registry.lookup",'
        f'"tool_version_sha256":"{TOOL_VERSION}","value":"NEW42"}},'
    )
    # Keep long link/producer/receipt suffixes explicit but shared between the
    # OK and ERROR golden results.
    result_suffix = (
        '"links":[{"relation":"result_of",'
        f'"target_evidence_content_sha256":"{"2" * 64}",'
        f'"target_evidence_id":"evd_{"2" * 24}"}}],'
        f'"producer":{{"kind":"tool","producer_id":"code_registry.lookup",'
        f'"version_sha256":"{TOOL_VERSION}"}},"receipt":{{"byte_count":32,'
        f'"kind":"local_test_receipt","schema_version":1,"sha256":"{RECEIPT}"}},'
        '"schema_version":1,"trajectory_id":"trajectory-1","type":"tool_result"}'
    )
    ok_wire = ok_prefix + result_suffix
    error_wire = (
        '{"body":{"call_id":"call-1","error_code":"lookup_error",'
        '"fact_namespace":"project_registry","key":"project-alpha",'
        '"result_content_sha256":'
        '"8fe9e3f7e5067c170e2673446be20b8fe2c3d38dbe72efa0ed978f34bcac12ce",'
        '"status":"error","tool_name":"code_registry.lookup",'
        f'"tool_version_sha256":"{TOOL_VERSION}","value":null}},'
        f"{result_suffix}"
    )
    vectors = (
        (
            _call(),
            call_wire,
            "1a4fa6a17324900cc9e2e71f82b7069872aed399e5877a19b75e2ab726e5ca13",
        ),
        (
            _result(),
            ok_wire,
            "389085713704d0a255d671b4fff0126fe1bf3db70f6e00ffce09229bf09c72eb",
        ),
        (
            _result(status=provenance.ToolResultStatusV1.ERROR, value=None),
            error_wire,
            "5c64b4736005712f4b4ac88a7b4ef1809582f6cdd55a9cb08c4df29181587358",
        ),
    )
    for value, expected_wire, expected_hash in vectors:
        assert provenance.provenance_payload_wire_v1(value) == expected_wire
        assert provenance.provenance_payload_sha256_v1(value) == expected_hash


def test_every_verification_verdict_has_a_locked_golden_vector() -> None:
    pass_wire = (
        '{"body":{"evaluator_id":"code_registry.verifier",'
        f'"evaluator_version_sha256":"{EVALUATOR_VERSION}",'
        '"outcome_type":"claim_verification","verdict":"pass"},'
        '"links":[{"relation":"evaluates",'
        f'"target_evidence_content_sha256":"{"3" * 64}",'
        f'"target_evidence_id":"evd_{"3" * 24}"}}],'
        f'"producer":{{"kind":"evaluator","producer_id":"code_registry.verifier",'
        f'"version_sha256":"{EVALUATOR_VERSION}"}},"receipt":{{"byte_count":32,'
        f'"kind":"local_test_receipt","schema_version":1,"sha256":"{RECEIPT}"}},'
        '"schema_version":1,"trajectory_id":"trajectory-1",'
        '"type":"verification_outcome"}'
    )
    expected = (
        (
            provenance.VerificationVerdictV1.PASS,
            pass_wire,
            "429ff988bcd50576536cf2b665d47ddcaa459e03734c3ad52e628123ae0d0950",
        ),
        (
            provenance.VerificationVerdictV1.FAIL,
            pass_wire.replace('"verdict":"pass"', '"verdict":"fail"'),
            "74d5ddf97e011b71c5a2cfc88eb95a921b0fc0b6e7bdc6e74f7da224233244b3",
        ),
        (
            provenance.VerificationVerdictV1.UNKNOWN,
            pass_wire.replace('"verdict":"pass"', '"verdict":"unknown"'),
            "6b5f56fa1b448df18b43e3bb60a2fb42fb9ec4db26883480e4d76a8e9282b235",
        ),
    )
    for verdict, expected_wire, expected_hash in expected:
        value = _outcome(verdict)
        assert provenance.provenance_payload_wire_v1(value) == expected_wire
        assert provenance.provenance_payload_sha256_v1(value) == expected_hash


@pytest.mark.parametrize(
    ("value", "payload_type", "relation"),
    (
        (_call(), "tool_call", "triggered_by"),
        (_result(), "tool_result", "result_of"),
        (
            _result(status=provenance.ToolResultStatusV1.ERROR, value=None),
            "tool_result",
            "result_of",
        ),
        (_outcome(), "verification_outcome", "evaluates"),
        (
            _outcome(provenance.VerificationVerdictV1.FAIL),
            "verification_outcome",
            "evaluates",
        ),
    ),
)
def test_every_linked_payload_round_trips_canonically(
    value: provenance.ProvenancePayloadV1,
    payload_type: str,
    relation: str,
) -> None:
    wire = provenance.provenance_payload_wire_v1(value)
    decoded = json.loads(wire)
    assert decoded["type"] == payload_type
    assert decoded["links"][0]["relation"] == relation
    assert provenance.parse_provenance_payload_v1(wire) == value


@pytest.mark.parametrize(
    ("kind", "value"),
    (
        ("feedback", _claim()),
        ("user_message", _claim()),
        ("tool_call", _call()),
        ("tool_result", _result()),
        ("outcome", _outcome()),
    ),
)
def test_evidence_parser_binds_kind_to_payload_role(
    kind: str,
    value: provenance.ProvenancePayloadV1,
) -> None:
    wire = provenance.provenance_payload_wire_v1(value)
    assert provenance.parse_evidence_provenance_payload_v1(kind, wire) == value


@pytest.mark.parametrize(
    ("kind", "value"),
    (
        ("environment", _claim()),
        ("feedback", _result()),
        ("tool_call", _outcome()),
        ("tool_result", _call()),
        ("outcome", _claim()),
        (object(), _result()),
    ),
)
def test_evidence_parser_rejects_cross_role_payload_smuggling(
    kind: object,
    value: provenance.ProvenancePayloadV1,
) -> None:
    wire = provenance.provenance_payload_wire_v1(value)
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.parse_evidence_provenance_payload_v1(kind, wire)  # type: ignore[arg-type]


def test_request_and_result_hashes_bind_every_logical_field() -> None:
    call = _call().body
    result = _result().body
    assert type(call) is provenance.ToolCallBodyV1
    assert type(result) is provenance.ToolResultBodyV1
    assert call.request_content_sha256 == provenance.tool_request_content_sha256_v1(
        tool_name=call.tool_name,
        tool_version_sha256=call.tool_version_sha256,
        fact_namespace=call.fact_namespace,
        key=call.key,
    )
    assert result.result_content_sha256 == provenance.tool_result_content_sha256_v1(
        call_id=result.call_id,
        tool_name=result.tool_name,
        tool_version_sha256=result.tool_version_sha256,
        status=result.status,
        fact_namespace=result.fact_namespace,
        key=result.key,
        value=result.value,
        error_code=result.error_code,
    )
    assert call.request_content_sha256 != provenance.tool_request_content_sha256_v1(
        tool_name=call.tool_name,
        tool_version_sha256=call.tool_version_sha256,
        fact_namespace=call.fact_namespace,
        key="project-beta",
    )
    assert result.result_content_sha256 != provenance.tool_result_content_sha256_v1(
        call_id=result.call_id,
        tool_name=result.tool_name,
        tool_version_sha256=result.tool_version_sha256,
        status=result.status,
        fact_namespace=result.fact_namespace,
        key=result.key,
        value="OTHER",
        error_code=result.error_code,
    )


def test_request_content_hash_deduplicates_calls_but_payload_hash_does_not() -> None:
    first = _call()
    first_body = first.body
    assert type(first_body) is provenance.ToolCallBodyV1
    second_body = dataclasses.replace(first_body, call_id="call-2")
    second = dataclasses.replace(first, body=second_body)
    assert first_body.request_content_sha256 == second_body.request_content_sha256
    assert provenance.provenance_payload_sha256_v1(first) != (
        provenance.provenance_payload_sha256_v1(second)
    )


def test_error_code_is_part_of_exact_result_content() -> None:
    timeout = _result(
        status=provenance.ToolResultStatusV1.ERROR,
        value=None,
        error_code="timeout",
    )
    not_found = _result(
        status=provenance.ToolResultStatusV1.ERROR,
        value=None,
        error_code="not_found",
    )
    timeout_body = timeout.body
    not_found_body = not_found.body
    assert type(timeout_body) is provenance.ToolResultBodyV1
    assert type(not_found_body) is provenance.ToolResultBodyV1
    assert timeout_body.result_content_sha256 != not_found_body.result_content_sha256


@pytest.mark.parametrize(
    ("status", "value", "error_code"),
    (
        (provenance.ToolResultStatusV1.OK, None, None),
        (provenance.ToolResultStatusV1.OK, "value", "unexpected_error"),
        (provenance.ToolResultStatusV1.ERROR, "unexpected", "lookup_error"),
        (provenance.ToolResultStatusV1.ERROR, None, None),
    ),
)
def test_result_status_and_value_presence_are_consistent(
    status: provenance.ToolResultStatusV1,
    value: str | None,
    error_code: str | None,
) -> None:
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.tool_result_content_sha256_v1(
            call_id="call-1",
            tool_name="code_registry.lookup",
            tool_version_sha256=TOOL_VERSION,
            status=status,
            fact_namespace="project_registry",
            key="project-alpha",
            value=value,
            error_code=error_code,
        )


def test_body_hash_mismatch_is_rejected_at_construction() -> None:
    call = _call().body
    result = _result().body
    assert type(call) is provenance.ToolCallBodyV1
    assert type(result) is provenance.ToolResultBodyV1
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(call, request_content_sha256="f" * 64)
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(result, result_content_sha256="f" * 64)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: dataclasses.replace(
            value,
            producer=dataclasses.replace(
                value.producer,
                kind=provenance.ProducerKindV1.USER,
                version_sha256=None,
            ),
        ),
        lambda value: dataclasses.replace(value, links=()),
        lambda value: dataclasses.replace(value, receipt=None),
        lambda value: dataclasses.replace(
            value,
            producer=dataclasses.replace(value.producer, producer_id="other.tool"),
        ),
    ),
)
def test_tool_result_requires_exact_producer_link_and_receipt(mutator: object) -> None:
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        mutator(_result())  # type: ignore[operator]


def test_outcome_requires_exact_evaluator_link_and_receipt() -> None:
    value = _outcome()
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(value, receipt=None)
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(
            value,
            links=(_link(provenance.ProvenanceRelationV1.RESULT_OF),),
        )
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(
            value,
            producer=dataclasses.replace(value.producer, producer_id="other.verifier"),
        )


def test_claim_cannot_smuggle_a_link_or_non_user_producer() -> None:
    value = _claim()
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(
            value,
            links=(_link(provenance.ProvenanceRelationV1.TRIGGERED_BY),),
        )
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(
            value,
            producer=provenance.ProvenanceProducerV1(
                kind=provenance.ProducerKindV1.AGENT,
                producer_id="agent",
                version_sha256=AGENT_VERSION,
            ),
        )


def test_producer_versions_and_receipts_have_one_canonical_presence_shape() -> None:
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(_claim().producer, version_sha256=AGENT_VERSION)
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(_call().producer, version_sha256=None)
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(_claim(), receipt=_receipt())
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(_call(), receipt=_receipt())


@pytest.mark.parametrize(
    "overrides",
    (
        {"kind": "bad kind"},
        {"schema_version": 0},
        {"schema_version": True},
        {"byte_count": -1},
        {"byte_count": True},
        {"sha256": "A" * 64},
    ),
)
def test_opaque_receipt_commitment_has_exact_reproducible_metadata(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "kind": "local_test_receipt",
        "schema_version": 1,
        "byte_count": 32,
        "sha256": RECEIPT,
    }
    values.update(overrides)
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.OpaqueReceiptCommitmentV1(**values)  # type: ignore[arg-type]


def test_link_requires_content_addressed_id_to_match_full_hash() -> None:
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.ProvenanceLinkV1(
            relation=provenance.ProvenanceRelationV1.RESULT_OF,
            target_evidence_id=f"evd_{'1' * 24}",
            target_evidence_content_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda wire: " " + wire,
        lambda wire: wire + "\n",
        lambda wire: wire.replace('"schema_version":1', '"schema_version":true'),
        lambda wire: wire.replace('"type":"fact_claim"', '"type":"unknown"'),
        lambda wire: wire.replace('"links":[]', '"links":{},"extra":[]'),
        lambda wire: wire.replace(
            '"trajectory_id":"trajectory-1"',
            '"trajectory_id":"trajectory-1","trajectory_id":"trajectory-2"',
        ),
        lambda wire: wire.replace('"receipt":null', '"receipt":NaN'),
    ),
)
def test_parser_rejects_noncanonical_ambiguous_or_unknown_json(
    mutation: object,
) -> None:
    wire = provenance.provenance_payload_wire_v1(_claim())
    with pytest.raises(provenance.LocalUpdateProvenanceError) as raised:
        provenance.parse_provenance_payload_v1(mutation(wire))  # type: ignore[operator]
    assert raised.value.reason == "closed_schema"
    assert str(raised.value) == "closed_schema"


def test_parser_revalidates_embedded_request_and_result_hashes() -> None:
    for value, field in (
        (_call(), "request_content_sha256"),
        (_result(), "result_content_sha256"),
    ):
        wire = provenance.provenance_payload_wire_v1(value)
        decoded = json.loads(wire)
        decoded["body"][field] = "f" * 64
        mutant = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with pytest.raises(
            provenance.LocalUpdateProvenanceError, match="closed_schema"
        ):
            provenance.parse_provenance_payload_v1(mutant)


def test_valid_unicode_is_canonicalized_and_invalid_unicode_is_rejected() -> None:
    value = dataclasses.replace(
        _claim(),
        body=provenance.FactClaimBodyV1(
            fact_namespace="project_registry",
            key="项目-甲",
            value="值-β",
        ),
    )
    wire = provenance.provenance_payload_wire_v1(value)
    assert "项目" not in wire
    assert provenance.parse_provenance_payload_v1(wire) == value

    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.FactClaimBodyV1(
            fact_namespace="project_registry",
            key="bad\ud800",
            value="value",
        )


def test_fact_identity_uses_nfc_printable_keys_and_explicit_value_rules() -> None:
    nfc = provenance.FactClaimBodyV1(
        fact_namespace="project_registry",
        key="café",
        value="line one\nline two\t",
    )
    assert nfc.key == "café"
    assert nfc.value.endswith("\t")
    empty = dataclasses.replace(nfc, value="")
    assert empty.value == ""

    for key in (
        " café",
        "café ",
        "cafe\u0301",
        "bad\x00key",
        "bad\nkey",
        "a\u2028b",
        "a\u2029b",
        "a\u00a0b",
        "a\u2007b",
        "a\u202fb",
    ):
        with pytest.raises(
            provenance.LocalUpdateProvenanceError,
            match="closed_schema",
        ):
            dataclasses.replace(nfc, key=key)
    for value in ("cafe\u0301", "bad\x00value"):
        with pytest.raises(
            provenance.LocalUpdateProvenanceError,
            match="closed_schema",
        ):
            dataclasses.replace(nfc, value=value)


def test_size_limits_and_exact_container_types_fail_closed() -> None:
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.FactClaimBodyV1(
            fact_namespace="project_registry",
            key="k" * 1025,
            value="value",
        )
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.FactClaimBodyV1(
            fact_namespace="project_registry",
            key="key",
            value="v" * (64 * 1024 + 1),
        )
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        dataclasses.replace(_claim(), links=[])
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.parse_provenance_payload_v1("x" * (128 * 1024 + 1))


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("trajectory_id",), "bad trajectory"),
        (("producer", "producer_id"), "bad producer"),
        (("body", "request_content_sha256"), "f" * 64),
        (("links", 0, "target_evidence_content_sha256"), "f" * 64),
    ),
)
def test_serialization_revalidates_low_level_mutation(
    path: tuple[object, ...],
    replacement: str,
) -> None:
    value = _call()
    target: object = value
    for component in path[:-1]:
        target = (
            target[component] if type(component) is int else getattr(target, component)
        )  # type: ignore[index]
    object.__setattr__(target, path[-1], replacement)
    with pytest.raises(provenance.LocalUpdateProvenanceError, match="closed_schema"):
        provenance.provenance_payload_wire_v1(value)


def test_payload_hash_binds_links_producer_receipt_and_body() -> None:
    original = _result()
    original_hash = provenance.provenance_payload_sha256_v1(original)
    mutants = (
        dataclasses.replace(
            original,
            links=(_link(provenance.ProvenanceRelationV1.RESULT_OF, digit="4"),),
        ),
        dataclasses.replace(
            original,
            receipt=dataclasses.replace(_receipt(), sha256="e" * 64),
        ),
        _result(status=provenance.ToolResultStatusV1.ERROR, value=None),
    )
    assert (
        len(
            {
                original_hash,
                *(provenance.provenance_payload_sha256_v1(v) for v in mutants),
            }
        )
        == 4
    )


def test_declared_producer_and_receipt_are_not_misnamed_as_authentication() -> None:
    """The wire carries commitments, but no secret or signature verification."""

    value = _result()
    wire = provenance.provenance_payload_wire_v1(value)
    assert '"producer_id":"code_registry.lookup"' in wire
    assert f'"sha256":"{RECEIPT}"' in wire
    assert "signature" not in wire
    assert "authenticated" not in wire
