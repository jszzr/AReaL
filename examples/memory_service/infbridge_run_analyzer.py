# SPDX-License-Identifier: Apache-2.0

"""Recover the model-boundary dry run from one sealed InfBridge ledger.

This module deliberately stops at :class:`ModelDryRunResult`.  The model-call
ledger does not contain actual Memory reader audits, provenance observations,
process-isolation witnesses, or cross-scope leakage sentinels.  Those facts
must come from a separate full-isolation observation artifact before callers
pass the recovered dry run to ``scoped_codebook_eval.analyze_model_run``.
Synthesizing those observations from preregistered expectations would turn an
expected contract into fake experimental evidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedDecoderTokenizer,
    InfBridgeModelAdapterError,
    RunEnvelopeV2,
    audited_model_call_execution_v2_from_artifacts,
)
from examples.memory_service.infbridge_run_ledger import (
    RunLedgerError,
    load_run_ledger,
)

from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

__all__ = [
    "InfBridgeRunAnalyzerError",
    "LedgerModelAttritionV1",
    "RecoveredModelDryRunV1",
    "recover_infbridge_model_dry_run_v1",
]


class InfBridgeRunAnalyzerError(RuntimeError):
    """Closed reason for refusing to reinterpret a valid ledger as a dry run.

    Storage, schema, and first-pass artifact-integrity failures intentionally
    remain :class:`RunLedgerError`; this type covers valid ledger states that
    have no honest dry-run projection and post-load replay-environment drift.
    """

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("analyzer error reason must be a non-empty str")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class LedgerModelAttritionV1:
    """The ledger's precise reason behind one evaluator-level missing call."""

    slot_index: int
    case_index: int
    arm: str
    reason: str


@dataclass(frozen=True, slots=True)
class RecoveredModelDryRunV1:
    """In-memory binding between a sealed run root and its model dry run.

    This is not a portable report format and carries no authenticity claim.
    ``ledger_attrition`` preserves the runtime-specific reason while the
    evaluator-compatible dry run intentionally projects every attrited slot to
    ``model_call_failure``.
    """

    schema_version: int
    run_id: str
    run_root_sha256: str
    manifest_sha256: str
    run_envelope_sha256: str
    seal_kind: str
    succeeded_count: int
    attrition_count: int
    dry_run: helpfulness.ModelDryRunResult
    ledger_attrition: tuple[LedgerModelAttritionV1, ...]


def recover_infbridge_model_dry_run_v1(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
) -> RecoveredModelDryRunV1:
    """Strictly replay one sealed ledger into the frozen evaluator call shape.

    ``complete`` and ``complete_with_attrition`` are recoverable.  ``OPEN``
    means execution is unfinished, while ``indeterminate`` cannot be honestly
    represented as the evaluator's known ``model_call_failure`` attrition and
    is therefore rejected.
    """

    snapshot = load_run_ledger(database_path, manifest, tokenizer, envelope)
    if snapshot.status != "SEALED":
        raise InfBridgeRunAnalyzerError("run_not_sealed")
    if snapshot.seal_kind == "indeterminate":
        raise InfBridgeRunAnalyzerError("run_indeterminate")
    if snapshot.seal_kind not in ("complete", "complete_with_attrition"):
        raise InfBridgeRunAnalyzerError("run_seal_kind")
    if (
        snapshot.stored_run_root_sha256 is None
        or snapshot.stored_run_root_sha256 != snapshot.computed_run_root_sha256
    ):
        raise RunLedgerError("ledger_corruption")

    calls: list[helpfulness.ModelDryRunCall] = []
    invalid_calls: list[helpfulness.ModelDryRunInvalidCall] = []
    ledger_attrition: list[LedgerModelAttritionV1] = []
    succeeded_count = 0
    backend = SGLangBridgeBackend()
    for slot_index, slot in enumerate(snapshot.slots):
        plan = slot.plan
        if plan.slot_index != slot_index or plan != envelope.call_plans[slot_index]:
            raise RunLedgerError("ledger_corruption")
        if slot.state == "SUCCEEDED":
            if (
                slot.receipt_bytes is None
                or slot.trace_bytes is None
                or slot.response_evidence_bytes is None
                or slot.decoded_response_utf8 is None
            ):
                raise RunLedgerError("ledger_corruption")
            try:
                execution = audited_model_call_execution_v2_from_artifacts(
                    manifest,
                    tokenizer,
                    envelope,
                    receipt_bytes=slot.receipt_bytes,
                    trace_bytes=slot.trace_bytes,
                    response_evidence_bytes=slot.response_evidence_bytes,
                    decoded_response_utf8=slot.decoded_response_utf8,
                    backend=backend,
                )
            except InfBridgeModelAdapterError as error:
                raise InfBridgeRunAnalyzerError("artifact_replay") from error
            calls.append(
                helpfulness.ModelDryRunCall(
                    case_index=plan.case_index,
                    arm=plan.arm,
                    attempt_index=0,
                    execution=execution.legacy_execution,
                )
            )
            succeeded_count += 1
            continue
        if slot.state == "ATTRITION" and slot.terminal_reason is not None:
            calls.append(
                helpfulness.ModelDryRunCall(
                    case_index=plan.case_index,
                    arm=plan.arm,
                    attempt_index=0,
                    execution=None,
                )
            )
            invalid_calls.append(
                helpfulness.ModelDryRunInvalidCall(
                    case_index=plan.case_index,
                    arm=plan.arm,
                    attempted=True,
                    reason="model_call_failure",
                )
            )
            ledger_attrition.append(
                LedgerModelAttritionV1(
                    slot_index=slot_index,
                    case_index=plan.case_index,
                    arm=plan.arm,
                    reason=slot.terminal_reason,
                )
            )
            continue
        raise RunLedgerError("ledger_corruption")

    dry_run = helpfulness.ModelDryRunResult(
        validity="invalid" if invalid_calls else "valid",
        invalid_calls=tuple(invalid_calls),
        manifest_sha256=snapshot.manifest_sha256,
        calls=tuple(calls),
    )
    if len(calls) != snapshot.call_count:
        raise RunLedgerError("ledger_corruption")
    return RecoveredModelDryRunV1(
        schema_version=1,
        run_id=snapshot.run_id,
        run_root_sha256=snapshot.stored_run_root_sha256,
        manifest_sha256=snapshot.manifest_sha256,
        run_envelope_sha256=snapshot.run_envelope_sha256,
        seal_kind=snapshot.seal_kind,
        succeeded_count=succeeded_count,
        attrition_count=len(ledger_attrition),
        dry_run=dry_run,
        ledger_attrition=tuple(ledger_attrition),
    )
