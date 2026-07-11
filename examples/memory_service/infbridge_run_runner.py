# SPDX-License-Identifier: Apache-2.0

"""Single-cursor, crash-conservative runner for the Memory model ledger.

At-most-once coordination is scoped to one explicit, trusted local directory.
The READY binding pins one run id to one database path and inode; it detects
ordinary replacement but not malicious ABA races, storage rollback, or another
coordination directory or host.  A crash before READY leaves a durable
reservation that fails closed and requires operator recovery.  The tests cover
process crashes and SQLite durability settings, not a physical power-loss
guarantee or remote-service idempotence.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import stat
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Literal, Protocol

from examples.memory_service import scoped_codebook_eval as helpfulness
from examples.memory_service.infbridge_model_adapter import (
    AuditedDecoderTokenizer,
    AuditedModelCallExecutionV2,
    InfBridgeModelAdapterError,
    RunEnvelopeV2,
    audited_model_call_receipt_v2_bytes,
)
from examples.memory_service.infbridge_run_ledger import (
    _ATTRITION_REASONS,
    RunLedgerError,
    RunLedgerSessionV1,
    RunLedgerSnapshotV1,
    _canonical_json_bytes,
    _fsync_parent_directory,
    _require_private_regular_database,
    _run_identity,
    _snapshot_database_path,
)

from areal.v2.inference_service.client_trace import (
    generation_physical_trace_bytes,
    generation_response_evidence_bytes,
)

__all__ = [
    "AuditedRunAdapter",
    "prepare_infbridge_ledger",
    "run_infbridge_ledger",
]

RunMode = Literal["new", "resume"]


class AuditedRunAdapter(Protocol):
    async def submit(
        self,
        case_index: int,
        arm: str,
    ) -> AuditedModelCallExecutionV2: ...


_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCK_PID = os.getpid()
_PROCESS_LOCK_PATHS: set[str] = set()
_TERMINAL_RETRY_COUNT = 4
_TERMINAL_RETRY_DELAY_SECONDS = 0.05


def _task_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _same_owner(file_stat: os.stat_result) -> bool:
    return not hasattr(os, "geteuid") or file_stat.st_uid == os.geteuid()


def _private_regular_file(file_stat: os.stat_result) -> bool:
    return (
        stat.S_ISREG(file_stat.st_mode)
        and file_stat.st_nlink == 1
        and file_stat.st_mode & 0o077 == 0
        and _same_owner(file_stat)
    )


def _snapshot_coordination_directory(
    coordination_directory: str | os.PathLike[str],
) -> str:
    try:
        raw_path = os.fspath(coordination_directory)
    except TypeError as error:
        raise RunLedgerError("ledger_coordination") from error
    if not isinstance(raw_path, str):
        raise RunLedgerError("ledger_coordination")
    path = str.__str__(raw_path)
    if (
        type(path) is not str
        or not path.strip()
        or "\x00" in path
        or path.startswith("//")
    ):
        raise RunLedgerError("ledger_coordination")
    try:
        path.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise RunLedgerError("ledger_coordination") from error
    absolute = os.path.abspath(path)
    parent = os.path.realpath(os.path.dirname(absolute))
    name = os.path.basename(absolute)
    if not name:
        raise RunLedgerError("ledger_coordination")
    return os.path.join(parent, name)


def _open_coordination_directory(
    coordination_directory: str | os.PathLike[str],
) -> tuple[str, int]:
    path = _snapshot_coordination_directory(coordination_directory)
    created = False
    descriptor: int | None = None
    try:
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        path_stat = os.lstat(path)
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or path_stat.st_mode & 0o077
            or not _same_owner(path_stat)
            or os.path.realpath(path) != path
        ):
            raise RunLedgerError("ledger_coordination")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(descriptor_stat.st_mode)
            or descriptor_stat.st_dev != path_stat.st_dev
            or descriptor_stat.st_ino != path_stat.st_ino
            or descriptor_stat.st_mode & 0o077
            or not _same_owner(descriptor_stat)
        ):
            os.close(descriptor)
            descriptor = None
            raise RunLedgerError("ledger_coordination")
        if created:
            os.fsync(descriptor)
            _fsync_parent_directory(path)
        return path, descriptor
    except RunLedgerError as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if error.reason == "ledger_coordination":
            raise
        raise RunLedgerError("ledger_coordination") from error
    except OSError as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise RunLedgerError("ledger_coordination") from error


@asynccontextmanager
async def _exclusive_process_lock(lock_path: str) -> AsyncIterator[None]:
    global _PROCESS_LOCK_PID

    claimed = False
    try:
        while not claimed:
            with _PROCESS_LOCK_GUARD:
                current_pid = os.getpid()
                if _PROCESS_LOCK_PID != current_pid:
                    _PROCESS_LOCK_PATHS.clear()
                    _PROCESS_LOCK_PID = current_pid
                if lock_path not in _PROCESS_LOCK_PATHS:
                    _PROCESS_LOCK_PATHS.add(lock_path)
                    claimed = True
            if not claimed:
                await asyncio.sleep(_TERMINAL_RETRY_DELAY_SECONDS)
        yield
    finally:
        if claimed:
            with _PROCESS_LOCK_GUARD:
                _PROCESS_LOCK_PATHS.discard(lock_path)


@asynccontextmanager
async def _exclusive_run_lock(
    coordination_descriptor: int,
    lock_name: str,
) -> AsyncIterator[None]:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    created = False
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                lock_name,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=coordination_descriptor,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(
                lock_name,
                flags,
                dir_fd=coordination_descriptor,
            )
        lock_stat = os.fstat(descriptor)
        if not _private_regular_file(lock_stat) or lock_stat.st_size != 0:
            raise RunLedgerError("ledger_lock")
        if created:
            os.fsync(descriptor)
            os.fsync(coordination_descriptor)
    except RunLedgerError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise RunLedgerError("ledger_lock") from error
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
            except OSError as error:
                raise RunLedgerError("ledger_lock") from error
        locked_stat = os.fstat(descriptor)
        path_stat = os.stat(
            lock_name,
            dir_fd=coordination_descriptor,
            follow_symlinks=False,
        )
        if (
            locked_stat.st_dev != lock_stat.st_dev
            or locked_stat.st_ino != lock_stat.st_ino
            or not _private_regular_file(locked_stat)
            or locked_stat.st_size != 0
            or path_stat.st_dev != locked_stat.st_dev
            or path_stat.st_ino != locked_stat.st_ino
        ):
            raise RunLedgerError("ledger_lock")
        try:
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _coordination_reservation_bytes(run_id: str, database_path: str) -> bytes:
    return _canonical_json_bytes(
        {
            "database_path": database_path,
            "kind": "areal-memory-run-coordination-reservation-v1",
            "run_id": run_id,
            "schema_version": 1,
        }
    )


def _coordination_binding_bytes(
    run_id: str,
    database_path: str,
    database_file_identity: tuple[int, int],
) -> bytes:
    return _canonical_json_bytes(
        {
            "database_file_device": database_file_identity[0],
            "database_file_inode": database_file_identity[1],
            "database_path": database_path,
            "kind": "areal-memory-run-coordination-v1",
            "run_id": run_id,
            "schema_version": 1,
        }
    )


def _binding_name(run_id: str) -> str:
    return f"{run_id}.json"


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("coordination binding write made no progress")
        offset += written


def _create_exact_coordination_file(
    coordination_descriptor: int,
    name: str,
    expected: bytes,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            flags,
            0o600,
            dir_fd=coordination_descriptor,
        )
        _write_all(descriptor, expected)
        file_stat = os.fstat(descriptor)
        if not _private_regular_file(file_stat) or file_stat.st_size != len(expected):
            raise RunLedgerError("ledger_coordination")
        os.fsync(descriptor)
        path_stat = os.stat(
            name,
            dir_fd=coordination_descriptor,
            follow_symlinks=False,
        )
        if path_stat.st_dev != file_stat.st_dev or path_stat.st_ino != file_stat.st_ino:
            raise RunLedgerError("ledger_coordination")
        os.fsync(coordination_descriptor)
    except FileExistsError as error:
        raise RunLedgerError("ledger_coordination") from error
    except RunLedgerError:
        raise
    except OSError as error:
        raise RunLedgerError("ledger_coordination") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _require_exact_coordination_file(
    coordination_descriptor: int,
    name: str,
    expected: bytes,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=coordination_descriptor)
        file_stat = os.fstat(descriptor)
        if not _private_regular_file(file_stat) or file_stat.st_size != len(expected):
            raise RunLedgerError("ledger_coordination")
        value = bytearray()
        while len(value) <= len(expected):
            chunk = os.read(descriptor, len(expected) + 1 - len(value))
            if not chunk:
                break
            value.extend(chunk)
        path_stat = os.stat(
            name,
            dir_fd=coordination_descriptor,
            follow_symlinks=False,
        )
        if (
            bytes(value) != expected
            or path_stat.st_dev != file_stat.st_dev
            or path_stat.st_ino != file_stat.st_ino
        ):
            raise RunLedgerError("ledger_coordination")
    except RunLedgerError:
        raise
    except OSError as error:
        raise RunLedgerError("ledger_coordination") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _create_coordination_reservation(
    coordination_descriptor: int,
    run_id: str,
    database_path: str,
) -> None:
    _create_exact_coordination_file(
        coordination_descriptor,
        _binding_name(run_id),
        _coordination_reservation_bytes(run_id, database_path),
    )


def _finalize_coordination_binding(
    coordination_descriptor: int,
    run_id: str,
    database_path: str,
    database_file_identity: tuple[int, int],
) -> None:
    binding_name = _binding_name(run_id)
    reservation = _coordination_reservation_bytes(run_id, database_path)
    ready = _coordination_binding_bytes(
        run_id,
        database_path,
        database_file_identity,
    )
    ready_name = f"{run_id}.ready"
    try:
        _require_exact_coordination_file(
            coordination_descriptor,
            binding_name,
            reservation,
        )
        if _require_private_regular_database(database_path) != database_file_identity:
            raise RunLedgerError("ledger_coordination")
        _create_exact_coordination_file(
            coordination_descriptor,
            ready_name,
            ready,
        )
        os.replace(
            ready_name,
            binding_name,
            src_dir_fd=coordination_descriptor,
            dst_dir_fd=coordination_descriptor,
        )
        os.fsync(coordination_descriptor)
        _require_exact_coordination_file(
            coordination_descriptor,
            binding_name,
            ready,
        )
        if _require_private_regular_database(database_path) != database_file_identity:
            raise RunLedgerError("ledger_coordination")
    except RunLedgerError as error:
        if error.reason == "ledger_coordination":
            raise
        raise RunLedgerError("ledger_coordination") from error
    except OSError as error:
        raise RunLedgerError("ledger_coordination") from error


def _require_coordination_binding(
    coordination_descriptor: int,
    run_id: str,
    database_path: str,
) -> tuple[int, int]:
    try:
        database_file_identity = _require_private_regular_database(database_path)
        expected = _coordination_binding_bytes(
            run_id,
            database_path,
            database_file_identity,
        )
        _require_exact_coordination_file(
            coordination_descriptor,
            _binding_name(run_id),
            expected,
        )
        if _require_private_regular_database(database_path) != database_file_identity:
            raise RunLedgerError("ledger_coordination")
        return database_file_identity
    except RunLedgerError as error:
        if error.reason == "ledger_coordination":
            raise
        raise RunLedgerError("ledger_coordination") from error


@asynccontextmanager
async def _coordinated_run(
    coordination_directory: str | os.PathLike[str],
    run_id: str,
    database_path: str,
    mode: RunMode,
) -> AsyncIterator[tuple[int, tuple[int, int] | None]]:
    directory_path, directory_descriptor = _open_coordination_directory(
        coordination_directory
    )
    lock_name = f"{run_id}.lock"
    lock_path = os.path.join(directory_path, lock_name)
    try:
        async with _exclusive_process_lock(lock_path):
            async with _exclusive_run_lock(directory_descriptor, lock_name):
                if mode == "new":
                    _create_coordination_reservation(
                        directory_descriptor,
                        run_id,
                        database_path,
                    )
                    database_file_identity = None
                else:
                    database_file_identity = _require_coordination_binding(
                        directory_descriptor,
                        run_id,
                        database_path,
                    )
                yield directory_descriptor, database_file_identity
    finally:
        try:
            os.close(directory_descriptor)
        except OSError:
            pass


def _terminal_matches(
    snapshot: RunLedgerSnapshotV1,
    slot_index: int,
    expected_state: Literal["SUCCEEDED", "ATTRITION"],
    *,
    attrition_reason: str | None,
    execution: AuditedModelCallExecutionV2 | None,
) -> bool:
    if snapshot.status != "OPEN":
        return False
    slot = snapshot.slots[slot_index]
    if slot.state != expected_state:
        return False
    if expected_state == "ATTRITION":
        return slot.terminal_reason == attrition_reason
    if execution is None:
        return False
    try:
        return (
            slot.receipt_bytes == audited_model_call_receipt_v2_bytes(execution.receipt)
            and slot.trace_bytes == generation_physical_trace_bytes(execution.trace)
            and slot.response_evidence_bytes
            == generation_response_evidence_bytes(execution.response_evidence)
            and slot.decoded_response_utf8
            == execution.response.encode("utf-8", errors="strict")
        )
    except Exception:
        return False


async def _retry_after_remote_call(
    session: RunLedgerSessionV1,
    slot_index: int,
    delay: float = _TERMINAL_RETRY_DELAY_SECONDS,
) -> None:
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError as cancellation:
        try:
            await _seal_indeterminate_durably(
                session,
                slot_index,
                "runner_cancelled",
                allow_lost_started=True,
            )
        except Exception as seal_error:
            cancellation.add_note(
                "cancellation sealing failed without replacing cancellation: "
                f"{type(seal_error).__name__}: {seal_error}"
            )
        raise


async def _seal_indeterminate_durably(
    session: RunLedgerSessionV1,
    slot_index: int,
    reason: str,
    *,
    allow_lost_started: bool,
) -> RunLedgerSnapshotV1:
    first_error: BaseException | None = None
    for attempt in range(_TERMINAL_RETRY_COUNT):
        try:
            return session.seal_indeterminate(slot_index, reason)
        except RunLedgerError as persistence_error:
            if first_error is None:
                first_error = persistence_error
        try:
            snapshot = session.refresh()
        except RunLedgerError as refresh_error:
            assert first_error is not None
            first_error.add_note(
                "indeterminate readback failed: "
                f"{type(refresh_error).__name__}: {refresh_error}"
            )
        else:
            slot = snapshot.slots[slot_index]
            if (
                snapshot.status == "SEALED"
                and snapshot.seal_kind == "indeterminate"
                and slot.state == "INDETERMINATE"
                and slot.terminal_reason == reason
            ):
                return snapshot
            if allow_lost_started and slot.state == "PLANNED":
                return session.seal_after_lost_started(slot_index)
            if slot.state != "STARTED":
                assert first_error is not None
                first_error.add_note(
                    "indeterminate readback found incompatible state "
                    f"{snapshot.status}/{slot.state}/{slot.terminal_reason}"
                )
                raise first_error
        if attempt + 1 < _TERMINAL_RETRY_COUNT:
            await asyncio.sleep(_TERMINAL_RETRY_DELAY_SECONDS)
    assert first_error is not None
    raise first_error


async def _mark_started_durably(
    session: RunLedgerSessionV1,
    slot_index: int,
) -> RunLedgerSnapshotV1:
    first_error: BaseException | None = None
    for attempt in range(_TERMINAL_RETRY_COUNT):
        try:
            return session.mark_started(slot_index)
        except RunLedgerError as persistence_error:
            if first_error is None:
                first_error = persistence_error
        try:
            snapshot = session.refresh()
        except RunLedgerError as refresh_error:
            assert first_error is not None
            first_error.add_note(
                "STARTED readback failed: "
                f"{type(refresh_error).__name__}: {refresh_error}"
            )
        else:
            slot = snapshot.slots[slot_index]
            if snapshot.status == "OPEN" and slot.state == "STARTED":
                return snapshot
            if snapshot.status != "OPEN" or slot.state != "PLANNED":
                assert first_error is not None
                first_error.add_note(
                    "STARTED readback found incompatible state "
                    f"{snapshot.status}/{slot.state}"
                )
                raise first_error
        if attempt + 1 < _TERMINAL_RETRY_COUNT:
            await asyncio.sleep(_TERMINAL_RETRY_DELAY_SECONDS)
    assert first_error is not None
    raise first_error


async def _persist_terminal(
    session: RunLedgerSessionV1,
    slot_index: int,
    expected_state: Literal["SUCCEEDED", "ATTRITION"],
    operation: Callable[[], RunLedgerSnapshotV1],
    *,
    attrition_reason: str | None = None,
    execution: AuditedModelCallExecutionV2 | None = None,
) -> RunLedgerSnapshotV1:
    first_error: BaseException | None = None
    for attempt in range(_TERMINAL_RETRY_COUNT):
        try:
            return operation()
        except RunLedgerError as persistence_error:
            if persistence_error.reason == "ledger_artifact":
                try:
                    await _seal_indeterminate_durably(
                        session,
                        slot_index,
                        "artifact_validation_failure",
                        allow_lost_started=True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as seal_error:
                    persistence_error.add_note(
                        "artifact-failure sealing failed: "
                        f"{type(seal_error).__name__}: {seal_error}"
                    )
                raise
            if first_error is None:
                first_error = persistence_error
        try:
            snapshot = session.refresh()
        except RunLedgerError as refresh_error:
            assert first_error is not None
            first_error.add_note(
                "terminal readback failed: "
                f"{type(refresh_error).__name__}: {refresh_error}"
            )
        else:
            if _terminal_matches(
                snapshot,
                slot_index,
                expected_state,
                attrition_reason=attrition_reason,
                execution=execution,
            ):
                return snapshot
            slot = snapshot.slots[slot_index]
            if slot.state == "PLANNED":
                assert first_error is not None
                try:
                    await _seal_indeterminate_durably(
                        session,
                        slot_index,
                        "started_record_lost",
                        allow_lost_started=True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as seal_error:
                    first_error.add_note(
                        "lost-STARTED sealing failed: "
                        f"{type(seal_error).__name__}: {seal_error}"
                    )
                raise first_error
            if slot.state != "STARTED":
                assert first_error is not None
                first_error.add_note(
                    "terminal readback found incompatible state "
                    f"{snapshot.status}/{slot.state}/{slot.terminal_reason}"
                )
                raise first_error
        if attempt + 1 < _TERMINAL_RETRY_COUNT:
            await _retry_after_remote_call(session, slot_index)
    assert first_error is not None
    try:
        await _seal_indeterminate_durably(
            session,
            slot_index,
            "terminal_persistence_failure",
            allow_lost_started=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception as seal_error:
        first_error.add_note(
            "terminal-persistence sealing failed: "
            f"{type(seal_error).__name__}: {seal_error}"
        )
    raise first_error


async def _seal_complete_durably(
    session: RunLedgerSessionV1,
) -> RunLedgerSnapshotV1:
    first_error: BaseException | None = None
    for attempt in range(_TERMINAL_RETRY_COUNT):
        try:
            return session.seal_complete()
        except RunLedgerError as persistence_error:
            if first_error is None:
                first_error = persistence_error
        try:
            snapshot = session.refresh()
        except RunLedgerError as refresh_error:
            assert first_error is not None
            first_error.add_note(
                "completion readback failed: "
                f"{type(refresh_error).__name__}: {refresh_error}"
            )
        else:
            if snapshot.status == "SEALED" and snapshot.seal_kind in (
                "complete",
                "complete_with_attrition",
            ):
                return snapshot
            if snapshot.status != "OPEN" or any(
                slot.state not in ("SUCCEEDED", "ATTRITION") for slot in snapshot.slots
            ):
                assert first_error is not None
                raise first_error
        if attempt + 1 < _TERMINAL_RETRY_COUNT:
            await asyncio.sleep(_TERMINAL_RETRY_DELAY_SECONDS)
    assert first_error is not None
    raise first_error


async def prepare_infbridge_ledger(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    *,
    coordination_directory: str | os.PathLike[str],
) -> RunLedgerSnapshotV1:
    """Create the run binding and all 384 PLANNED slots without model calls."""

    path = _snapshot_database_path(database_path)
    identity = _run_identity(manifest, tokenizer, envelope)
    async with _coordinated_run(
        coordination_directory,
        identity.run_id,
        path,
        "new",
    ) as (coordination_descriptor, _bound_file_identity):
        session = RunLedgerSessionV1.create(
            path,
            manifest,
            tokenizer,
            envelope,
        )
        if session.snapshot.run_id != identity.run_id:
            raise RunLedgerError("ledger_identity")
        _finalize_coordination_binding(
            coordination_descriptor,
            identity.run_id,
            path,
            session.database_file_identity,
        )
        return session.snapshot


async def run_infbridge_ledger(
    database_path: str | os.PathLike[str],
    manifest: helpfulness.ModelRunManifest,
    tokenizer: AuditedDecoderTokenizer,
    envelope: RunEnvelopeV2,
    adapter: AuditedRunAdapter,
    *,
    mode: RunMode,
    coordination_directory: str | os.PathLike[str],
) -> RunLedgerSnapshotV1:
    """Run one fixed ledger in an explicit local run-id coordination domain.

    A logical slot is marked STARTED in a durable transaction before the
    adapter is invoked.  A recovered STARTED slot is never reissued: it is
    sealed INDETERMINATE because the prior process may have reached the remote
    service.  InfBridge may still perform multiple physical abort/resubmit
    attempts inside that one logical adapter invocation.
    """

    if type(mode) is not str or mode not in ("new", "resume"):
        raise RunLedgerError("ledger_mode")
    path = _snapshot_database_path(database_path)
    identity = _run_identity(manifest, tokenizer, envelope)
    async with _coordinated_run(
        coordination_directory,
        identity.run_id,
        path,
        mode,
    ) as (coordination_descriptor, bound_file_identity):
        session = (
            RunLedgerSessionV1.create(path, manifest, tokenizer, envelope)
            if mode == "new"
            else RunLedgerSessionV1.resume(path, manifest, tokenizer, envelope)
        )
        if session.snapshot.run_id != identity.run_id:
            raise RunLedgerError("ledger_identity")
        if mode == "new":
            _finalize_coordination_binding(
                coordination_descriptor,
                identity.run_id,
                path,
                session.database_file_identity,
            )
        else:
            if session.database_file_identity != bound_file_identity:
                raise RunLedgerError("ledger_coordination")
            if (
                _require_coordination_binding(
                    coordination_descriptor,
                    identity.run_id,
                    path,
                )
                != bound_file_identity
            ):
                raise RunLedgerError("ledger_coordination")
        snapshot = session.snapshot
        if snapshot.status == "SEALED":
            return snapshot
        started = tuple(
            slot.plan.slot_index for slot in snapshot.slots if slot.state == "STARTED"
        )
        if started:
            return await _seal_indeterminate_durably(
                session,
                started[0],
                "orphan_started",
                allow_lost_started=False,
            )
        while True:
            planned = next(
                (slot for slot in session.snapshot.slots if slot.state == "PLANNED"),
                None,
            )
            if planned is None:
                return await _seal_complete_durably(session)
            slot_index = planned.plan.slot_index
            await _mark_started_durably(session, slot_index)
            try:
                execution = await adapter.submit(
                    planned.plan.case_index,
                    planned.plan.arm,
                )
                if _task_cancelling():
                    raise asyncio.CancelledError(
                        "adapter returned after cancellation was requested"
                    )
            except InfBridgeModelAdapterError as adapter_error:
                if _task_cancelling():
                    cancellation = asyncio.CancelledError(
                        "adapter suppressed a pending cancellation"
                    )
                    try:
                        await _seal_indeterminate_durably(
                            session,
                            slot_index,
                            "runner_cancelled",
                            allow_lost_started=True,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as seal_error:
                        cancellation.add_note(
                            "cancellation sealing failed: "
                            f"{type(seal_error).__name__}: {seal_error}"
                        )
                    raise cancellation from adapter_error
                if adapter_error.reason not in _ATTRITION_REASONS:
                    try:
                        await _seal_indeterminate_durably(
                            session,
                            slot_index,
                            "unexpected_failure",
                            allow_lost_started=True,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as seal_error:
                        adapter_error.add_note(
                            "invalid-adapter-reason sealing failed: "
                            f"{type(seal_error).__name__}: {seal_error}"
                        )
                    raise
                attrition_reason = adapter_error.reason
                await _persist_terminal(
                    session,
                    slot_index,
                    "ATTRITION",
                    lambda: session.record_attrition(
                        slot_index,
                        attrition_reason,
                    ),
                    attrition_reason=attrition_reason,
                )
                continue
            except BaseException as execution_error:
                cancellation_pending = (
                    isinstance(execution_error, asyncio.CancelledError)
                    or _task_cancelling()
                )
                reason = (
                    "runner_cancelled" if cancellation_pending else "unexpected_failure"
                )
                try:
                    await _seal_indeterminate_durably(
                        session,
                        slot_index,
                        reason,
                        allow_lost_started=True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as seal_error:
                    execution_error.add_note(
                        "indeterminate sealing failed without replacing the "
                        "execution error: "
                        f"{type(seal_error).__name__}: {seal_error}"
                    )
                if cancellation_pending and not isinstance(
                    execution_error, asyncio.CancelledError
                ):
                    raise asyncio.CancelledError(
                        "adapter suppressed a pending cancellation"
                    ) from execution_error
                raise
            if type(execution) is not AuditedModelCallExecutionV2:
                artifact_error = RunLedgerError("ledger_artifact")
                try:
                    await _seal_indeterminate_durably(
                        session,
                        slot_index,
                        "artifact_validation_failure",
                        allow_lost_started=True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as seal_error:
                    artifact_error.add_note(
                        "invalid-execution sealing failed: "
                        f"{type(seal_error).__name__}: {seal_error}"
                    )
                raise artifact_error
            await _persist_terminal(
                session,
                slot_index,
                "SUCCEEDED",
                lambda: session.record_success(
                    slot_index,
                    execution,
                ),
                execution=execution,
            )
