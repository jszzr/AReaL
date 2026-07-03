# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.v2.cli.inference.client import RouterClient


def test_router_register_worker_sends_stable_incarnation_payload(monkeypatch):
    client = RouterClient("http://router", "admin")
    calls = []

    def fake_post(path, payload=None, *, timeout=10.0, auth=True):
        calls.append((path, payload, timeout, auth))
        return {"status": "ok", "worker_id": payload["worker_id"]}

    monkeypatch.setattr(client, "_post", fake_post)

    first = client.register_worker(
        "http://proxy:5001", "proxy-incarnation-1", None, timeout=3.0
    )
    second = client.register_worker(
        "http://proxy:5001", "proxy-incarnation-1", None, timeout=3.0
    )

    expected_payload = {
        "worker_addr": "http://proxy:5001",
        "worker_id": "proxy-incarnation-1",
        "expected_worker_id": None,
    }
    assert calls == [
        ("/register", expected_payload, 3.0, True),
        ("/register", expected_payload, 3.0, True),
    ]
    assert first["worker_id"] == second["worker_id"] == "proxy-incarnation-1"


def test_router_unregister_worker_sends_exact_incarnation_payload(monkeypatch):
    client = RouterClient("http://router", "admin")
    calls = []

    def fake_post(path, payload=None, *, timeout=10.0, auth=True):
        calls.append((path, payload, timeout, auth))
        return {"status": "ok", "removed": True}

    monkeypatch.setattr(client, "_post", fake_post)

    response = client.unregister_worker(
        "http://proxy:5001", "proxy-incarnation-1", timeout=4.0
    )

    assert calls == [
        (
            "/unregister",
            {
                "worker_addr": "http://proxy:5001",
                "worker_id": "proxy-incarnation-1",
            },
            4.0,
            True,
        )
    ]
    assert response["removed"] is True
