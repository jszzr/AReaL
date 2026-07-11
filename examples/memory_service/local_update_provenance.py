# SPDX-License-Identifier: Apache-2.0

"""Canonical provenance payloads for local Memory update experiments.

The core Memory Service intentionally stores an evidence payload as an opaque
UTF-8 string.  This module defines one opt-in, backwards-compatible profile for
experiments that need to join a fact claim to a tool call, its result, and an
independent verification outcome.  The canonical JSON remains inside the
existing :class:`EvidenceEvent` payload, so the existing event content hash and
evidence snapshot commit to every field without a database migration.

The links bind both an evidence ID and its full event content hash.  A later
store-backed projector can verify snapshot membership, ID/full-hash equality,
and recorded ingest precedence.  None of those facts authenticate a producer.
``producer_id`` and an opaque receipt commitment remain declarations until a
trusted writer or signature-verifying ingress binds them.  Policies must keep
that trust boundary explicit.

Fact identities use exact NFC-normalized Unicode.  Keys are non-blank,
printable, and have no surrounding whitespace; values may be empty and may use
tabs or newlines, but reject every other Unicode ``C*`` category character.  A tool
request content hash deliberately excludes ``call_id`` so identical requests
can be recognized across attempts.  The enclosing payload hash still binds the
call instance.  A result content hash includes ``call_id`` and, on failure, a
stable ``error_code``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "FactClaimBodyV1",
    "LocalUpdateProvenanceError",
    "OpaqueReceiptCommitmentV1",
    "ProducerKindV1",
    "ProvenanceLinkV1",
    "ProvenancePayloadV1",
    "ProvenanceProducerV1",
    "ProvenanceRelationV1",
    "ToolCallBodyV1",
    "ToolResultBodyV1",
    "ToolResultStatusV1",
    "VerificationOutcomeBodyV1",
    "VerificationVerdictV1",
    "parse_evidence_provenance_payload_v1",
    "parse_provenance_payload_v1",
    "provenance_payload_sha256_v1",
    "provenance_payload_wire_v1",
    "tool_request_content_sha256_v1",
    "tool_result_content_sha256_v1",
]


class LocalUpdateProvenanceError(ValueError):
    """Stable, payload-free reason for rejecting a provenance value."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("provenance error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


class ProducerKindV1(StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    EVALUATOR = "evaluator"


class ProvenanceRelationV1(StrEnum):
    TRIGGERED_BY = "triggered_by"
    RESULT_OF = "result_of"
    EVALUATES = "evaluates"


class ToolResultStatusV1(StrEnum):
    OK = "ok"
    ERROR = "error"


class VerificationVerdictV1(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


_SCHEMA_VERSION = 1
_MAX_PAYLOAD_UTF8_BYTES = 128 * 1024
_MAX_FACT_VALUE_UTF8_BYTES = 64 * 1024
_MAX_FACT_KEY_UTF8_BYTES = 1024
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_EVIDENCE_ID_PATTERN = re.compile(r"evd_[0-9a-f]{24}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUEST_HASH_DOMAIN = b"areal-memory-provenance-tool-request-v1\0"
_RESULT_HASH_DOMAIN = b"areal-memory-provenance-tool-result-v1\0"
_PAYLOAD_HASH_DOMAIN = b"areal-memory-provenance-payload-v1\0"


def _closed_schema(error: BaseException | None = None) -> LocalUpdateProvenanceError:
    result = LocalUpdateProvenanceError("closed_schema")
    if error is not None:
        result.__cause__ = error
    return result


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise _closed_schema(error)


def _bounded_text(
    value: object,
    *,
    allow_blank: bool,
    maximum_utf8_bytes: int,
) -> str:
    if type(value) is not str:
        raise _closed_schema()
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise _closed_schema(error)
    if (not allow_blank and not value.strip()) or len(encoded) > maximum_utf8_bytes:
        raise _closed_schema()
    return value


def _token(value: object) -> str:
    text = _bounded_text(value, allow_blank=False, maximum_utf8_bytes=128)
    if _TOKEN_PATTERN.fullmatch(text) is None:
        raise _closed_schema()
    return text


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise _closed_schema()
    return value


def _optional_sha256(value: object) -> str | None:
    if value is None:
        return None
    return _sha256(value)


def _non_negative_int64(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise _closed_schema()
    return value


def _evidence_id(value: object) -> str:
    if type(value) is not str or _EVIDENCE_ID_PATTERN.fullmatch(value) is None:
        raise _closed_schema()
    return value


def _fact_namespace(value: object) -> str:
    return _token(value)


def _fact_key(value: object) -> str:
    text = _bounded_text(
        value,
        allow_blank=False,
        maximum_utf8_bytes=_MAX_FACT_KEY_UTF8_BYTES,
    )
    if (
        text != unicodedata.normalize("NFC", text)
        or text != text.strip()
        or not text.isprintable()
        or any(unicodedata.category(character).startswith("C") for character in text)
    ):
        raise _closed_schema()
    return text


def _fact_value(value: object) -> str:
    text = _bounded_text(
        value,
        allow_blank=True,
        maximum_utf8_bytes=_MAX_FACT_VALUE_UTF8_BYTES,
    )
    if text != unicodedata.normalize("NFC", text) or any(
        unicodedata.category(character).startswith("C")
        and character not in {"\n", "\t"}
        for character in text
    ):
        raise _closed_schema()
    return text


def _enum_exact(value: object, enum_type: type[StrEnum]) -> StrEnum:
    if type(value) is not enum_type:
        raise _closed_schema()
    return value


@dataclass(frozen=True, slots=True)
class ProvenanceProducerV1:
    kind: ProducerKindV1
    producer_id: str
    version_sha256: str | None

    def __post_init__(self) -> None:
        kind = _enum_exact(self.kind, ProducerKindV1)
        producer_id = _token(self.producer_id)
        version_sha256 = _optional_sha256(self.version_sha256)
        if (kind is ProducerKindV1.USER) != (version_sha256 is None):
            raise _closed_schema()
        object.__setattr__(self, "producer_id", producer_id)
        object.__setattr__(self, "version_sha256", version_sha256)


@dataclass(frozen=True, slots=True)
class ProvenanceLinkV1:
    relation: ProvenanceRelationV1
    target_evidence_id: str
    target_evidence_content_sha256: str

    def __post_init__(self) -> None:
        _enum_exact(self.relation, ProvenanceRelationV1)
        target_evidence_id = _evidence_id(self.target_evidence_id)
        target_evidence_content_sha256 = _sha256(self.target_evidence_content_sha256)
        if target_evidence_id != f"evd_{target_evidence_content_sha256[:24]}":
            raise _closed_schema()
        object.__setattr__(
            self,
            "target_evidence_id",
            target_evidence_id,
        )
        object.__setattr__(
            self,
            "target_evidence_content_sha256",
            target_evidence_content_sha256,
        )


@dataclass(frozen=True, slots=True)
class OpaqueReceiptCommitmentV1:
    """Commitment to exact opaque receipt bytes, not proof of their authenticity."""

    kind: str
    schema_version: int
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _token(self.kind))
        schema_version = _non_negative_int64(self.schema_version)
        if schema_version == 0:
            raise _closed_schema()
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "byte_count", _non_negative_int64(self.byte_count))
        object.__setattr__(self, "sha256", _sha256(self.sha256))


@dataclass(frozen=True, slots=True)
class FactClaimBodyV1:
    fact_namespace: str
    key: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_namespace", _fact_namespace(self.fact_namespace))
        object.__setattr__(self, "key", _fact_key(self.key))
        object.__setattr__(self, "value", _fact_value(self.value))


def tool_request_content_sha256_v1(
    *,
    tool_name: str,
    tool_version_sha256: str,
    fact_namespace: str,
    key: str,
) -> str:
    """Hash request content; repeated call instances may share this hash."""

    value = {
        "fact_namespace": _fact_namespace(fact_namespace),
        "key": _fact_key(key),
        "schema_version": _SCHEMA_VERSION,
        "tool_name": _token(tool_name),
        "tool_version_sha256": _sha256(tool_version_sha256),
    }
    return hashlib.sha256(
        _REQUEST_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolCallBodyV1:
    call_id: str
    tool_name: str
    tool_version_sha256: str
    fact_namespace: str
    key: str
    request_content_sha256: str

    def __post_init__(self) -> None:
        call_id = _token(self.call_id)
        tool_name = _token(self.tool_name)
        tool_version_sha256 = _sha256(self.tool_version_sha256)
        fact_namespace = _fact_namespace(self.fact_namespace)
        key = _fact_key(self.key)
        request_content_sha256 = _sha256(self.request_content_sha256)
        expected = tool_request_content_sha256_v1(
            tool_name=tool_name,
            tool_version_sha256=tool_version_sha256,
            fact_namespace=fact_namespace,
            key=key,
        )
        if request_content_sha256 != expected:
            raise _closed_schema()
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_version_sha256", tool_version_sha256)
        object.__setattr__(self, "fact_namespace", fact_namespace)
        object.__setattr__(self, "key", key)
        object.__setattr__(
            self,
            "request_content_sha256",
            request_content_sha256,
        )


def tool_result_content_sha256_v1(
    *,
    call_id: str,
    tool_name: str,
    tool_version_sha256: str,
    status: ToolResultStatusV1,
    fact_namespace: str,
    key: str,
    value: str | None,
    error_code: str | None,
) -> str:
    """Hash exact result content, including its call ID and stable error code."""

    status = _enum_exact(status, ToolResultStatusV1)
    if value is None:
        normalized_value = None
    else:
        normalized_value = _fact_value(value)
    if error_code is None:
        normalized_error_code = None
    else:
        normalized_error_code = _token(error_code)
    if status is ToolResultStatusV1.OK:
        valid_shape = normalized_value is not None and normalized_error_code is None
    else:
        valid_shape = normalized_value is None and normalized_error_code is not None
    if not valid_shape:
        raise _closed_schema()
    body = {
        "call_id": _token(call_id),
        "error_code": normalized_error_code,
        "fact_namespace": _fact_namespace(fact_namespace),
        "key": _fact_key(key),
        "schema_version": _SCHEMA_VERSION,
        "status": status.value,
        "tool_name": _token(tool_name),
        "tool_version_sha256": _sha256(tool_version_sha256),
        "value": normalized_value,
    }
    return hashlib.sha256(_RESULT_HASH_DOMAIN + _canonical_json_bytes(body)).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolResultBodyV1:
    call_id: str
    tool_name: str
    tool_version_sha256: str
    status: ToolResultStatusV1
    fact_namespace: str
    key: str
    value: str | None
    error_code: str | None
    result_content_sha256: str

    def __post_init__(self) -> None:
        call_id = _token(self.call_id)
        tool_name = _token(self.tool_name)
        tool_version_sha256 = _sha256(self.tool_version_sha256)
        status = _enum_exact(self.status, ToolResultStatusV1)
        fact_namespace = _fact_namespace(self.fact_namespace)
        key = _fact_key(self.key)
        if self.value is None:
            value = None
        else:
            value = _fact_value(self.value)
        if self.error_code is None:
            error_code = None
        else:
            error_code = _token(self.error_code)
        result_content_sha256 = _sha256(self.result_content_sha256)
        expected = tool_result_content_sha256_v1(
            call_id=call_id,
            tool_name=tool_name,
            tool_version_sha256=tool_version_sha256,
            status=status,
            fact_namespace=fact_namespace,
            key=key,
            value=value,
            error_code=error_code,
        )
        if result_content_sha256 != expected:
            raise _closed_schema()
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_version_sha256", tool_version_sha256)
        object.__setattr__(self, "fact_namespace", fact_namespace)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "error_code", error_code)
        object.__setattr__(
            self,
            "result_content_sha256",
            result_content_sha256,
        )


@dataclass(frozen=True, slots=True)
class VerificationOutcomeBodyV1:
    outcome_type: str
    verdict: VerificationVerdictV1
    evaluator_id: str
    evaluator_version_sha256: str

    def __post_init__(self) -> None:
        outcome_type = _bounded_text(
            self.outcome_type,
            allow_blank=False,
            maximum_utf8_bytes=64,
        )
        if outcome_type != "claim_verification":
            raise _closed_schema()
        _enum_exact(self.verdict, VerificationVerdictV1)
        object.__setattr__(self, "outcome_type", outcome_type)
        object.__setattr__(self, "evaluator_id", _token(self.evaluator_id))
        object.__setattr__(
            self,
            "evaluator_version_sha256",
            _sha256(self.evaluator_version_sha256),
        )


ProvenanceBodyV1 = (
    FactClaimBodyV1 | ToolCallBodyV1 | ToolResultBodyV1 | VerificationOutcomeBodyV1
)


@dataclass(frozen=True, slots=True)
class ProvenancePayloadV1:
    schema_version: int
    trajectory_id: str
    producer: ProvenanceProducerV1
    links: tuple[ProvenanceLinkV1, ...]
    body: ProvenanceBodyV1
    receipt: OpaqueReceiptCommitmentV1 | None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise _closed_schema()
        trajectory_id = _token(self.trajectory_id)
        if type(self.producer) is not ProvenanceProducerV1:
            raise _closed_schema()
        if type(self.links) is not tuple or any(
            type(link) is not ProvenanceLinkV1 for link in self.links
        ):
            raise _closed_schema()
        if (
            self.receipt is not None
            and type(self.receipt) is not OpaqueReceiptCommitmentV1
        ):
            raise _closed_schema()

        body = self.body
        producer = self.producer
        if type(body) is FactClaimBodyV1:
            if (
                producer.kind is not ProducerKindV1.USER
                or self.links
                or self.receipt is not None
            ):
                raise _closed_schema()
        elif type(body) is ToolCallBodyV1:
            if (
                producer.kind is not ProducerKindV1.AGENT
                or len(self.links) != 1
                or self.links[0].relation is not ProvenanceRelationV1.TRIGGERED_BY
                or self.receipt is not None
            ):
                raise _closed_schema()
        elif type(body) is ToolResultBodyV1:
            if (
                producer.kind is not ProducerKindV1.TOOL
                or producer.producer_id != body.tool_name
                or producer.version_sha256 != body.tool_version_sha256
                or len(self.links) != 1
                or self.links[0].relation is not ProvenanceRelationV1.RESULT_OF
                or self.receipt is None
            ):
                raise _closed_schema()
        elif type(body) is VerificationOutcomeBodyV1:
            if (
                producer.kind is not ProducerKindV1.EVALUATOR
                or producer.producer_id != body.evaluator_id
                or producer.version_sha256 != body.evaluator_version_sha256
                or len(self.links) != 1
                or self.links[0].relation is not ProvenanceRelationV1.EVALUATES
                or self.receipt is None
            ):
                raise _closed_schema()
        else:
            raise _closed_schema()

        object.__setattr__(self, "trajectory_id", trajectory_id)


def _producer_value(value: ProvenanceProducerV1) -> dict[str, object]:
    if type(value) is not ProvenanceProducerV1:
        raise _closed_schema()
    return {
        "kind": value.kind.value,
        "producer_id": value.producer_id,
        "version_sha256": value.version_sha256,
    }


def _link_value(value: ProvenanceLinkV1) -> dict[str, object]:
    if type(value) is not ProvenanceLinkV1:
        raise _closed_schema()
    return {
        "relation": value.relation.value,
        "target_evidence_content_sha256": value.target_evidence_content_sha256,
        "target_evidence_id": value.target_evidence_id,
    }


def _receipt_value(value: OpaqueReceiptCommitmentV1) -> dict[str, object]:
    if type(value) is not OpaqueReceiptCommitmentV1:
        raise _closed_schema()
    return {
        "byte_count": value.byte_count,
        "kind": value.kind,
        "schema_version": value.schema_version,
        "sha256": value.sha256,
    }


def _body_value(value: ProvenanceBodyV1) -> tuple[str, dict[str, object]]:
    if type(value) is FactClaimBodyV1:
        return (
            "fact_claim",
            {
                "fact_namespace": value.fact_namespace,
                "key": value.key,
                "value": value.value,
            },
        )
    if type(value) is ToolCallBodyV1:
        return (
            "tool_call",
            {
                "call_id": value.call_id,
                "fact_namespace": value.fact_namespace,
                "key": value.key,
                "request_content_sha256": value.request_content_sha256,
                "tool_name": value.tool_name,
                "tool_version_sha256": value.tool_version_sha256,
            },
        )
    if type(value) is ToolResultBodyV1:
        return (
            "tool_result",
            {
                "call_id": value.call_id,
                "error_code": value.error_code,
                "fact_namespace": value.fact_namespace,
                "key": value.key,
                "result_content_sha256": value.result_content_sha256,
                "status": value.status.value,
                "tool_name": value.tool_name,
                "tool_version_sha256": value.tool_version_sha256,
                "value": value.value,
            },
        )
    if type(value) is VerificationOutcomeBodyV1:
        return (
            "verification_outcome",
            {
                "evaluator_id": value.evaluator_id,
                "evaluator_version_sha256": value.evaluator_version_sha256,
                "outcome_type": value.outcome_type,
                "verdict": value.verdict.value,
            },
        )
    raise _closed_schema()


def _validated_payload_copy(value: ProvenancePayloadV1) -> ProvenancePayloadV1:
    """Revalidate a frozen DTO at each serialization boundary.

    ``frozen=True`` prevents ordinary assignment, but it is not an authenticity
    capability.  Rebuilding every nested value also rejects objects modified via
    low-level ``object.__setattr__`` after construction.
    """

    if type(value) is not ProvenancePayloadV1:
        raise _closed_schema()
    producer = value.producer
    if type(producer) is not ProvenanceProducerV1:
        raise _closed_schema()
    normalized_producer = ProvenanceProducerV1(
        kind=producer.kind,
        producer_id=producer.producer_id,
        version_sha256=producer.version_sha256,
    )
    if type(value.links) is not tuple:
        raise _closed_schema()
    normalized_link_items: list[ProvenanceLinkV1] = []
    for link in value.links:
        if type(link) is not ProvenanceLinkV1:
            raise _closed_schema()
        normalized_link_items.append(
            ProvenanceLinkV1(
                relation=link.relation,
                target_evidence_id=link.target_evidence_id,
                target_evidence_content_sha256=link.target_evidence_content_sha256,
            )
        )
    normalized_links = tuple(normalized_link_items)
    body = value.body
    if type(body) is FactClaimBodyV1:
        normalized_body: ProvenanceBodyV1 = FactClaimBodyV1(
            fact_namespace=body.fact_namespace,
            key=body.key,
            value=body.value,
        )
    elif type(body) is ToolCallBodyV1:
        normalized_body = ToolCallBodyV1(
            call_id=body.call_id,
            tool_name=body.tool_name,
            tool_version_sha256=body.tool_version_sha256,
            fact_namespace=body.fact_namespace,
            key=body.key,
            request_content_sha256=body.request_content_sha256,
        )
    elif type(body) is ToolResultBodyV1:
        normalized_body = ToolResultBodyV1(
            call_id=body.call_id,
            tool_name=body.tool_name,
            tool_version_sha256=body.tool_version_sha256,
            status=body.status,
            fact_namespace=body.fact_namespace,
            key=body.key,
            value=body.value,
            error_code=body.error_code,
            result_content_sha256=body.result_content_sha256,
        )
    elif type(body) is VerificationOutcomeBodyV1:
        normalized_body = VerificationOutcomeBodyV1(
            outcome_type=body.outcome_type,
            verdict=body.verdict,
            evaluator_id=body.evaluator_id,
            evaluator_version_sha256=body.evaluator_version_sha256,
        )
    else:
        raise _closed_schema()
    receipt = value.receipt
    if receipt is None:
        normalized_receipt = None
    elif type(receipt) is OpaqueReceiptCommitmentV1:
        normalized_receipt = OpaqueReceiptCommitmentV1(
            kind=receipt.kind,
            schema_version=receipt.schema_version,
            byte_count=receipt.byte_count,
            sha256=receipt.sha256,
        )
    else:
        raise _closed_schema()
    normalized = ProvenancePayloadV1(
        schema_version=value.schema_version,
        trajectory_id=value.trajectory_id,
        producer=normalized_producer,
        links=normalized_links,
        body=normalized_body,
        receipt=normalized_receipt,
    )
    if normalized != value:
        raise _closed_schema()
    return normalized


def _payload_value(value: ProvenancePayloadV1) -> dict[str, object]:
    value = _validated_payload_copy(value)
    payload_type, body = _body_value(value.body)
    return {
        "body": body,
        "links": [_link_value(link) for link in value.links],
        "producer": _producer_value(value.producer),
        "receipt": None if value.receipt is None else _receipt_value(value.receipt),
        "schema_version": value.schema_version,
        "trajectory_id": value.trajectory_id,
        "type": payload_type,
    }


def provenance_payload_wire_v1(value: ProvenancePayloadV1) -> str:
    """Return the exact canonical JSON text stored in ``EvidenceEvent.payload``."""

    canonical = _canonical_json_bytes(_payload_value(value))
    if len(canonical) > _MAX_PAYLOAD_UTF8_BYTES:
        raise _closed_schema()
    return canonical.decode("ascii")


def provenance_payload_sha256_v1(value: ProvenancePayloadV1) -> str:
    """Hash the typed payload independently from its enclosing evidence event."""

    return hashlib.sha256(
        _PAYLOAD_HASH_DOMAIN + provenance_payload_wire_v1(value).encode("ascii")
    ).hexdigest()


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _exact_object(value: object, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise _closed_schema()
    return value


def _parse_producer(value: object) -> ProvenanceProducerV1:
    obj = _exact_object(
        value,
        frozenset(("kind", "producer_id", "version_sha256")),
    )
    try:
        kind = ProducerKindV1(obj["kind"])
    except (TypeError, ValueError) as error:
        raise _closed_schema(error)
    return ProvenanceProducerV1(
        kind=kind,
        producer_id=obj["producer_id"],  # type: ignore[arg-type]
        version_sha256=obj["version_sha256"],  # type: ignore[arg-type]
    )


def _parse_link(value: object) -> ProvenanceLinkV1:
    obj = _exact_object(
        value,
        frozenset(
            (
                "relation",
                "target_evidence_content_sha256",
                "target_evidence_id",
            )
        ),
    )
    try:
        relation = ProvenanceRelationV1(obj["relation"])
    except (TypeError, ValueError) as error:
        raise _closed_schema(error)
    return ProvenanceLinkV1(
        relation=relation,
        target_evidence_id=obj["target_evidence_id"],  # type: ignore[arg-type]
        target_evidence_content_sha256=obj["target_evidence_content_sha256"],  # type: ignore[arg-type]
    )


def _parse_receipt(value: object) -> OpaqueReceiptCommitmentV1 | None:
    if value is None:
        return None
    obj = _exact_object(
        value,
        frozenset(("byte_count", "kind", "schema_version", "sha256")),
    )
    return OpaqueReceiptCommitmentV1(
        kind=obj["kind"],  # type: ignore[arg-type]
        schema_version=obj["schema_version"],  # type: ignore[arg-type]
        byte_count=obj["byte_count"],  # type: ignore[arg-type]
        sha256=obj["sha256"],  # type: ignore[arg-type]
    )


def _parse_body(payload_type: object, value: object) -> ProvenanceBodyV1:
    if payload_type == "fact_claim":
        obj = _exact_object(value, frozenset(("fact_namespace", "key", "value")))
        return FactClaimBodyV1(
            fact_namespace=obj["fact_namespace"],  # type: ignore[arg-type]
            key=obj["key"],  # type: ignore[arg-type]
            value=obj["value"],  # type: ignore[arg-type]
        )
    if payload_type == "tool_call":
        obj = _exact_object(
            value,
            frozenset(
                (
                    "call_id",
                    "fact_namespace",
                    "key",
                    "request_content_sha256",
                    "tool_name",
                    "tool_version_sha256",
                )
            ),
        )
        return ToolCallBodyV1(
            call_id=obj["call_id"],  # type: ignore[arg-type]
            tool_name=obj["tool_name"],  # type: ignore[arg-type]
            tool_version_sha256=obj["tool_version_sha256"],  # type: ignore[arg-type]
            fact_namespace=obj["fact_namespace"],  # type: ignore[arg-type]
            key=obj["key"],  # type: ignore[arg-type]
            request_content_sha256=obj["request_content_sha256"],  # type: ignore[arg-type]
        )
    if payload_type == "tool_result":
        obj = _exact_object(
            value,
            frozenset(
                (
                    "call_id",
                    "error_code",
                    "fact_namespace",
                    "key",
                    "result_content_sha256",
                    "status",
                    "tool_name",
                    "tool_version_sha256",
                    "value",
                )
            ),
        )
        try:
            status = ToolResultStatusV1(obj["status"])
        except (TypeError, ValueError) as error:
            raise _closed_schema(error)
        return ToolResultBodyV1(
            call_id=obj["call_id"],  # type: ignore[arg-type]
            tool_name=obj["tool_name"],  # type: ignore[arg-type]
            tool_version_sha256=obj["tool_version_sha256"],  # type: ignore[arg-type]
            status=status,
            fact_namespace=obj["fact_namespace"],  # type: ignore[arg-type]
            key=obj["key"],  # type: ignore[arg-type]
            value=obj["value"],  # type: ignore[arg-type]
            error_code=obj["error_code"],  # type: ignore[arg-type]
            result_content_sha256=obj["result_content_sha256"],  # type: ignore[arg-type]
        )
    if payload_type == "verification_outcome":
        obj = _exact_object(
            value,
            frozenset(
                (
                    "evaluator_id",
                    "evaluator_version_sha256",
                    "outcome_type",
                    "verdict",
                )
            ),
        )
        try:
            verdict = VerificationVerdictV1(obj["verdict"])
        except (TypeError, ValueError) as error:
            raise _closed_schema(error)
        return VerificationOutcomeBodyV1(
            outcome_type=obj["outcome_type"],  # type: ignore[arg-type]
            verdict=verdict,
            evaluator_id=obj["evaluator_id"],  # type: ignore[arg-type]
            evaluator_version_sha256=obj["evaluator_version_sha256"],  # type: ignore[arg-type]
        )
    raise _closed_schema()


def parse_provenance_payload_v1(value: str) -> ProvenancePayloadV1:
    """Parse exact canonical V1 syntax without claiming an evidence role."""

    text = _bounded_text(
        value,
        allow_blank=False,
        maximum_utf8_bytes=_MAX_PAYLOAD_UTF8_BYTES,
    )
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (
        json.JSONDecodeError,
        _DuplicateKeyError,
        TypeError,
        ValueError,
        RecursionError,
    ) as error:
        raise _closed_schema(error)
    obj = _exact_object(
        decoded,
        frozenset(
            (
                "body",
                "links",
                "producer",
                "receipt",
                "schema_version",
                "trajectory_id",
                "type",
            )
        ),
    )
    links_wire = obj["links"]
    if type(links_wire) is not list:
        raise _closed_schema()
    try:
        result = ProvenancePayloadV1(
            schema_version=obj["schema_version"],  # type: ignore[arg-type]
            trajectory_id=obj["trajectory_id"],  # type: ignore[arg-type]
            producer=_parse_producer(obj["producer"]),
            links=tuple(_parse_link(link) for link in links_wire),
            body=_parse_body(obj["type"], obj["body"]),
            receipt=_parse_receipt(obj["receipt"]),
        )
    except LocalUpdateProvenanceError:
        raise
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as error:
        raise _closed_schema(error)
    if provenance_payload_wire_v1(result) != text:
        raise _closed_schema()
    return result


def parse_evidence_provenance_payload_v1(
    evidence_kind: str,
    value: str,
) -> ProvenancePayloadV1:
    """Parse a payload and bind its body type to an evidence-kind value.

    This remains a local value check.  A store-backed projector must separately
    prove that every link resolves to the promised ID/full hash, precedes its
    child at the snapshot ingest boundary, and has matching trajectory fields.

    The adapter accepts exact string values instead of importing AReaL so this
    canonical payload layer remains usable by a lightweight ingress process.
    """

    if type(evidence_kind) is not str:
        raise _closed_schema()
    payload = parse_provenance_payload_v1(value)
    expected_body_type: type[object]
    if evidence_kind in {"feedback", "user_message"}:
        expected_body_type = FactClaimBodyV1
    elif evidence_kind == "tool_call":
        expected_body_type = ToolCallBodyV1
    elif evidence_kind == "tool_result":
        expected_body_type = ToolResultBodyV1
    elif evidence_kind == "outcome":
        expected_body_type = VerificationOutcomeBodyV1
    else:
        raise _closed_schema()
    if type(payload.body) is not expected_body_type:
        raise _closed_schema()
    return payload
