import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self._payload = {
            "group_id": "group-1",
            "expected_version": 7,
            "sessions": [
                {
                    "session_id": "session-1",
                    "session_api_key": "opaque-session-key",
                }
            ],
        }
        self.text = json.dumps(self._payload)

    def json(self) -> dict[str, object]:
        return self._payload


@pytest.fixture
def start_session_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3] / "examples" / "hermes" / "start_session.py"
    )
    spec = importlib.util.spec_from_file_location("hermes_start_session", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    outcomes: list[object],
    *,
    request_id: str = "logical-start-1",
    capacity_timeout: str = "10",
    monotonic_values: list[float] | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    pending = iter(outcomes)

    def post(url: str, **kwargs: object) -> object:
        calls.append({"url": url, **kwargs})
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(module.requests, "post", post)
    if monotonic_values is None:
        monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    else:
        monotonic = iter(monotonic_values)
        monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "start_session.py",
            "http://gateway",
            "--admin-key",
            "admin",
            "--request-id",
            request_id,
            "--capacity-timeout",
            capacity_timeout,
        ],
    )

    module.main()
    return calls


def test_prints_request_id_and_omits_unsupported_refresh_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    start_session_module: ModuleType,
) -> None:
    calls = _run_main(monkeypatch, start_session_module, [_Response(201)])

    captured = capsys.readouterr()
    assert "logical-start-1" in captured.out
    assert "REQUEST_ID=logical-start-1" in captured.err
    assert calls[0]["json"] == {
        "task_id": "demo-task",
        "delivery_mode": "callback",
        "request_id": "logical-start-1",
        "request_expires_at": 1_010.0,
    }


@pytest.mark.parametrize(
    "transient",
    [
        _Response(429),
        _Response(503),
        pytest.param("transport", id="transport-error"),
    ],
)
def test_retries_transient_failure_with_same_request_id(
    monkeypatch: pytest.MonkeyPatch,
    start_session_module: ModuleType,
    transient: object,
) -> None:
    if transient == "transport":
        transient = start_session_module.requests.ConnectionError("connection reset")

    calls = _run_main(
        monkeypatch,
        start_session_module,
        [transient, _Response(201)],
    )

    assert len(calls) == 2
    assert [call["json"]["request_id"] for call in calls] == [
        "logical-start-1",
        "logical-start-1",
    ]
    assert [call["json"]["request_expires_at"] for call in calls] == [
        1_010.0,
        1_010.0,
    ]


def test_does_not_retry_non_transient_4xx(
    monkeypatch: pytest.MonkeyPatch,
    start_session_module: ModuleType,
) -> None:
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, start_session_module, [_Response(422), _Response(201)])


def test_does_not_start_another_attempt_at_retry_deadline(
    monkeypatch: pytest.MonkeyPatch,
    start_session_module: ModuleType,
) -> None:
    with pytest.raises(SystemExit):
        _run_main(
            monkeypatch,
            start_session_module,
            [_Response(503), _Response(201)],
            capacity_timeout="0.1",
            monotonic_values=[0.0, 0.05, 0.1],
        )
