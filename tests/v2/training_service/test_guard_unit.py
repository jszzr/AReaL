"""Unit tests for training-service guard process handoff."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areal.v2.training_service.guard.app import _state, app


@pytest.fixture
def client(tmp_path):
    _state.server_host = "127.0.0.1"
    _state.fileroot = str(tmp_path)
    _state.experiment_name = "seed-test"
    _state.trial_name = "trial-0"
    _state.forked_children.clear()
    _state.forked_children_map.clear()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
    _state.forked_children.clear()
    _state.forked_children_map.clear()


@patch("areal.infra.rpc.guard.app.run_with_streaming_logs")
def test_guard_passes_seed_contract_to_new_worker_interpreter(mock_run, client):
    process = MagicMock(pid=42)
    process.poll.return_value = None
    mock_run.return_value = process
    raw_cmd = [
        "python",
        "-m",
        "areal.v2.training_service.worker",
        "--seed",
        "20260703",
        "--seed-role",
        "actor",
        "--seed-rank",
        "1",
    ]

    response = client.post(
        "/fork",
        json={"role": "train-worker", "worker_index": 1, "raw_cmd": raw_cmd},
    )

    assert response.status_code == 200
    assert mock_run.call_args.args[0] == raw_cmd
