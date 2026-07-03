"""Unit tests for training-service worker Flask app."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from tests.v2.training_service.fake_train_engine import FakeTrainEngine

from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import seeding
from areal.v2.training_service.worker.config import TrainWorkerConfig

MODULE = "areal.v2.training_service.worker.app"


@pytest.fixture(autouse=True)
def reset_worker_state():
    import areal.v2.training_service.worker.app as worker_app

    if worker_app._engine_work_queue is not None:
        worker_app._engine_work_queue.put(None)
    if worker_app._engine_thread is not None:
        worker_app._engine_thread.join(timeout=1.0)

    worker_app._engine = None
    worker_app._node_addr = ""
    worker_app._engine_thread = None
    worker_app._engine_work_queue = None

    yield

    if worker_app._engine_work_queue is not None:
        worker_app._engine_work_queue.put(None)
    if worker_app._engine_thread is not None:
        worker_app._engine_thread.join(timeout=1.0)

    worker_app._engine = None
    worker_app._node_addr = ""
    worker_app._engine_thread = None
    worker_app._engine_work_queue = None


@pytest.fixture
def client():
    from areal.v2.training_service.worker.app import create_app

    app = create_app(
        TrainWorkerConfig(
            host="127.0.0.1",
            port=19001,
            admin_api_key="worker-admin",
        )
    )
    return app.test_client()


class TestWorkerEngineCreation:
    def test_create_engine_requires_engine_class(self, client):
        resp = client.post(
            "/create_engine",
            json={"init_args": [], "init_kwargs": {}},
        )
        assert resp.status_code == 400
        assert "engine_class" in resp.get_json()["error"]

    def test_create_engine_success(self, client):
        resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["status"] == "success"

    def test_unseeded_worker_preserves_existing_engine_creation(self, client):
        with patch("areal.utils.seeding.set_random_seed") as set_seed:
            resp = client.post(
                "/create_engine",
                json={
                    "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                    "init_args": serialize_value([]),
                    "init_kwargs": serialize_value({"world_size": 1}),
                },
            )

        assert resp.status_code == 200
        set_seed.assert_not_called()

    def test_seed_is_set_in_engine_thread_before_engine_construction(self):
        from areal.v2.training_service.worker.app import create_app

        events = []

        class SeedOrderEngine(FakeTrainEngine):
            def __init__(self, *args, **kwargs):
                events.append(("engine",))
                super().__init__(*args, **kwargs)

        config = SimpleNamespace(
            host="127.0.0.1",
            port=19001,
            admin_api_key="worker-admin",
            log_level="warning",
            seed=20260703,
            seed_role="actor",
            seed_rank=0,
        )
        app = create_app(config)

        def record_seed(base_seed, key):
            events.append(("seed", base_seed, key))

        with (
            patch(
                "areal.v2.training_service.worker.engine.import_from_string",
                return_value=SeedOrderEngine,
            ),
            patch("areal.utils.seeding.set_random_seed", side_effect=record_seed),
        ):
            resp = app.test_client().post(
                "/create_engine",
                json={
                    "engine_class": "tests.SeedOrderEngine",
                    "init_args": serialize_value([]),
                    "init_kwargs": serialize_value({}),
                },
            )

        assert resp.status_code == 200
        assert events == [("seed", 20260703, "actor0"), ("engine",)]


def test_worker_cli_parses_seed_contract_into_config():
    from areal.v2.training_service.worker.__main__ import main

    app = MagicMock()
    config = SimpleNamespace(
        host="127.0.0.1",
        port=19001,
        admin_api_key="worker-admin",
        log_level="info",
    )

    with (
        patch.object(
            sys,
            "argv",
            [
                "train-worker",
                "--host",
                "127.0.0.1",
                "--port",
                "19001",
                "--admin-api-key",
                "worker-admin",
                "--log-level",
                "info",
                "--seed",
                "20260703",
                "--seed-role",
                "actor",
                "--seed-rank",
                "3",
            ],
        ),
        patch(
            "areal.v2.training_service.worker.config.TrainWorkerConfig",
            return_value=config,
        ) as config_cls,
        patch("areal.v2.training_service.worker.app.create_app", return_value=app),
        patch("areal.utils.logging.suppress_http_loggers"),
    ):
        main()

    config_cls.assert_called_once_with(
        host="127.0.0.1",
        port=19001,
        admin_api_key="worker-admin",
        log_level="info",
        seed=20260703,
        seed_role="actor",
        seed_rank=3,
    )
    app.run.assert_called_once_with(host="127.0.0.1", port=19001, threaded=True)


@pytest.mark.parametrize(
    "seed_kwargs",
    [
        {"seed": 1},
        {"seed_role": "actor"},
        {"seed_rank": 0},
        {"seed": 1, "seed_role": "actor"},
        {"seed": 1, "seed_rank": 0},
        {"seed_role": "actor", "seed_rank": 0},
    ],
)
def test_worker_seed_contract_is_all_or_none(seed_kwargs):
    with pytest.raises(ValueError, match="seed.*seed_role.*seed_rank.*together"):
        TrainWorkerConfig(**seed_kwargs)


@pytest.mark.parametrize("invalid_seed", [-1, 2**32, True])
def test_worker_seed_rejects_values_outside_uint32(invalid_seed):
    with pytest.raises(ValueError, match="seed.*unsigned 32-bit"):
        TrainWorkerConfig(
            seed=invalid_seed,
            seed_role="actor",
            seed_rank=0,
        )


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_worker_seed_accepts_uint32_boundaries(seed):
    config = TrainWorkerConfig(seed=seed, seed_role="actor", seed_rank=0)
    assert config.seed == seed


@pytest.mark.parametrize("invalid_role", ["", "   ", 1])
def test_worker_seed_rejects_invalid_role(invalid_role):
    with pytest.raises(ValueError, match="seed_role.*non-empty string"):
        TrainWorkerConfig(seed=1, seed_role=invalid_role, seed_rank=0)


@pytest.mark.parametrize("invalid_rank", [-1, True, "0"])
def test_worker_seed_rejects_invalid_rank(invalid_rank):
    with pytest.raises(ValueError, match="seed_rank.*non-negative integer"):
        TrainWorkerConfig(seed=1, seed_role="actor", seed_rank=invalid_rank)


def test_worker_cli_rejects_partial_seed_contract():
    from areal.v2.training_service.worker.__main__ import main

    with (
        patch.object(
            sys,
            "argv",
            [
                "train-worker",
                "--host",
                "127.0.0.1",
                "--admin-api-key",
                "worker-admin",
                "--seed",
                "7",
            ],
        ),
        patch(
            "areal.v2.training_service.worker.app.create_app",
            return_value=MagicMock(),
        ),
    ):
        with pytest.raises(ValueError, match="seed.*seed_role.*seed_rank.*together"):
            main()


class _TinyLoraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


def _lora_state(base_seed):
    seeding.set_random_seed(base_seed, key="actor0")
    model = get_peft_model(
        _TinyLoraModel(),
        LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]),
    )
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def test_same_worker_seed_reproduces_real_peft_lora_initialization():
    first = _lora_state(7)
    repeated = _lora_state(7)
    changed = _lora_state(8)

    assert first.keys() == repeated.keys() == changed.keys()
    assert all(torch.equal(first[name], repeated[name]) for name in first)

    lora_a_names = [name for name in first if "lora_A" in name]
    lora_b_names = [name for name in first if "lora_B" in name]
    assert any(not torch.equal(first[name], changed[name]) for name in lora_a_names)
    assert all(torch.count_nonzero(first[name]) > 0 for name in lora_a_names)
    assert all(torch.count_nonzero(first[name]) == 0 for name in lora_b_names)
    assert all(torch.count_nonzero(changed[name]) == 0 for name in lora_b_names)


class TestWorkerEndpoints:
    def test_topology_before_create_engine_returns_400(self, client):
        resp = client.get("/topology")
        assert resp.status_code == 400
        assert "Engine not created" in resp.get_json()["error"]

    def test_train_batch_after_create_engine(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        train_resp = client.post(
            "/train_batch",
            json={
                "args": serialize_value(
                    [{"token_ids": [1, 2, 3], "metadata": {"weight": 2.0}}]
                ),
                "kwargs": serialize_value({}),
            },
        )
        assert train_resp.status_code == 200
        result = deserialize_value(train_resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result

    def test_topology_after_create_engine(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        with patch.dict(
            "os.environ",
            {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"},
            clear=False,
        ):
            topo_resp = client.get("/topology")
        assert topo_resp.status_code == 200
        topo = topo_resp.get_json()
        assert topo["rank"] == 0
        assert topo["world_size"] == 1
        assert topo["dp_size"] == 1

    def test_ppo_endpoints_return_400_when_engine_method_missing(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        payload = {
            "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
            "kwargs": serialize_value({}),
        }
        for path in [
            "/ppo/actor/compute_logp",
            "/ppo/actor/compute_advantages",
            "/ppo/actor/update",
            "/ppo/critic/compute_values",
            "/ppo/critic/update",
        ]:
            resp = client.post(path, json=payload)
            assert resp.status_code == 400
            assert "does not implement method" in resp.get_json()["error"]

    def test_forward_batch_after_initialize_succeeds_without_distributed_group(
        self, client
    ):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        init_resp = client.post(
            "/initialize",
            json={
                "args": serialize_value([]),
                "kwargs": serialize_value({"addr": None, "ft_spec": None}),
            },
        )
        assert init_resp.status_code == 200

        forward_resp = client.post(
            "/forward_batch",
            json={
                "args": serialize_value(
                    [[{"token_ids": [1, 2, 3], "metadata": {"weight": 2.0}}]]
                ),
                "kwargs": serialize_value({"output_seqlens": [3]}),
            },
        )
        assert forward_resp.status_code == 200
        result = deserialize_value(forward_resp.get_json()["result"])
        assert isinstance(result, dict)
        assert result["output_seqlens"] == [3]

    def test_sft_route_succeeds_without_distributed_group_for_single_worker(
        self, client
    ):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        resp = client.post(
            "/sft/train",
            json={
                "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
                "kwargs": serialize_value({}),
            },
        )
        assert resp.status_code == 200
        result = deserialize_value(resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result

    def test_sft_route_ignores_rpc_meta_override_for_single_worker(self, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        resp = client.post(
            "/sft/train",
            json={
                "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
                "kwargs": serialize_value({}),
                "rpc_meta": {"broadcast": False},
            },
        )
        assert resp.status_code == 200
        result = deserialize_value(resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result
