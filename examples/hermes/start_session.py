#!/usr/bin/env python3
"""Start an RL session on AReaL's inference gateway.

Mints an opaque per-session key on the inference gateway that
``train.py`` embeds, then prints it. You forward that key to the Hermes
Agent Service (``hermes_loop.py``) so the agent's LLM calls flow through the
inference service under it and get captured as a training trajectory; you then
score it with ``set_reward.py``.

The inference gateway's admin key is whatever you passed as
``rollout.admin_api_key`` to ``train.py``.

Each invocation creates a new callback episode and a new opaque session key.
``request_id`` identifies that one logical creation: retries after a capacity
response, server failure, or transport failure reuse the same ID.

Usage:
    python start_session.py http://host:port --admin-key sk-xxx
    python start_session.py http://host:port --admin-key sk-xxx --task-id my-task
    python start_session.py http://host:port --admin-key sk-xxx --request-id run-1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import requests

# =============================================================================
# CLI formatting helpers (ANSI colors + output helpers)
# =============================================================================
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"


def info(msg: str) -> None:
    print(f"  {CYAN}ℹ{RESET}  {msg}")


def success(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")


def error(msg: str) -> None:
    print(f"  {RED}✘{RESET}  {msg}")


def arrow(msg: str) -> None:
    print(f"  {YELLOW}→{RESET} {msg}")


def die(msg: str) -> None:
    error(msg)
    sys.exit(1)


def show_request(method: str, path: str, auth_label: str, gateway_url: str) -> None:
    print(f"  {DIM}{method} {gateway_url}/{path}{RESET}")
    print(f"  {DIM}Auth: {auth_label}{RESET}")


def show_response(status_code: int, body: str) -> None:
    if 200 <= status_code < 300:
        print(f"  {GREEN}HTTP {status_code}{RESET}")
    else:
        print(f"  {RED}HTTP {status_code}{RESET}")
    if body:
        try:
            formatted = json.dumps(json.loads(body), indent=2)
            for line in formatted.split("\n"):
                print(f"  {DIM}{line}{RESET}")
        except (json.JSONDecodeError, ValueError):
            for line in body.split("\n"):
                print(f"  {DIM}{line}{RESET}")


def header(title: str) -> None:
    """Print a boxed header."""
    print()
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start an AReaL RL session")
    parser.add_argument("gateway_url", help="Inference gateway URL")
    parser.add_argument(
        "--admin-key",
        default=os.getenv("ADMIN_KEY", "areal-admin-key"),
        help="Inference gateway admin key (== rollout.admin_api_key; env: ADMIN_KEY)",
    )
    parser.add_argument(
        "--task-id",
        default=os.getenv("TASK_ID", "demo-task"),
        help="Task identifier (env: TASK_ID)",
    )
    parser.add_argument(
        "--request-id",
        default=os.getenv("REQUEST_ID"),
        help=(
            "Stable idempotency key for this logical session start "
            "(env: REQUEST_ID; generated once when omitted)"
        ),
    )
    parser.add_argument(
        "--capacity-timeout",
        type=float,
        default=300.0,
        help="Seconds to retry HTTP 429, transport errors, and HTTP 5xx",
    )
    args = parser.parse_args()
    request_id = args.request_id or f"hermes-{uuid.uuid4()}"

    header("Start Session")
    info("Requesting a new RL session (admin auth → gateway routes to a worker)")
    arrow(f"Request ID : {BOLD}{request_id}{RESET}")
    print(f"REQUEST_ID={request_id}", file=sys.stderr)
    show_request("POST", "rl/start_session", "Bearer ***", args.gateway_url)

    deadline = time.monotonic() + args.capacity_timeout
    resp: requests.Response | None = None
    last_error: requests.RequestException | None = None
    first_attempt = True
    while first_attempt or time.monotonic() < deadline:
        first_attempt = False
        try:
            resp = requests.post(
                f"{args.gateway_url}/rl/start_session",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {args.admin_key}",
                },
                json={
                    "task_id": args.task_id,
                    "delivery_mode": "callback",
                    "request_id": request_id,
                },
                timeout=10,
            )
        except requests.RequestException as e:
            resp = None
            last_error = e
        else:
            last_error = None

        retryable = last_error is not None or (
            resp is not None
            and (resp.status_code == 429 or 500 <= resp.status_code < 600)
        )
        if not retryable:
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if last_error is not None:
            info(
                f"Gateway transport error; retrying request {request_id}: {last_error}"
            )
        else:
            assert resp is not None
            info(
                f"Gateway returned HTTP {resp.status_code}; retrying request {request_id}"
            )
        time.sleep(min(0.5, remaining))

    if resp is None:
        die(f"Failed to reach gateway before retry deadline: {last_error}")

    show_response(resp.status_code, resp.text)

    if resp.status_code != 201:
        die(
            f"start_session failed (HTTP {resp.status_code}). "
            "HTTP 429 means the trainer did not grant capacity before the retry deadline."
        )

    # v2 inference gateway returns 201 with a flat list of session credentials:
    #   {"group_id": ..., "sessions": [{"session_id": ..., "session_api_key": ...}]}
    try:
        data = resp.json()
        session = data["sessions"][0]
        session_api_key = session["session_api_key"]
        session_id = session["session_id"]
    except (ValueError, KeyError, IndexError) as e:
        die(f"Failed to parse response: {e}")

    success("Session started!")
    arrow(f"Session ID : {BOLD}{session_id}{RESET}")
    arrow(f"API Key    : {BOLD}{session_api_key}{RESET}")
    print()
    info("Forward this key to the Hermes Agent Service to capture the trajectory:")
    print()
    print(
        f"  python hermes_loop.py <agent-gateway>"
        f" --admin-api-key <agent-admin-key>"
        f" --inf-base-url {args.gateway_url}"
        f" --session-api-key {session_api_key}"
    )
    print()
    info("Then score the episode with the same key:")
    print()
    print(
        f"  python set_reward.py {args.gateway_url}"
        f" --api-key {session_api_key} --reward 1.0"
    )
    print()
    info("To start the next episode, create a new session and opaque key:")
    print()
    print(f"  python start_session.py {args.gateway_url} --admin-key <admin-key>")
    print()

    # Machine-readable output on stderr for scripting
    print(f"SESSION_API_KEY={session_api_key}", file=sys.stderr)
    print(f"SESSION_ID={session_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
