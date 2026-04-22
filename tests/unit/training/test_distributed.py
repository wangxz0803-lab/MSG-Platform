"""Unit tests for distributed training helpers (single-process path)."""

from __future__ import annotations

import torch

from msg_embedding.training.distributed import (
    DistEnv,
    all_reduce_mean,
    barrier,
    distributed_context,
    setup_distributed,
    teardown_distributed,
)


def test_setup_distributed_single_process(monkeypatch) -> None:
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    env = setup_distributed()
    assert isinstance(env, DistEnv)
    assert env.world_size == 1
    assert env.rank == 0
    assert env.is_main is True
    assert env.is_distributed is False
    assert env.backend == "none"
    teardown_distributed(env)


def test_distributed_context_manager(monkeypatch) -> None:
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    with distributed_context() as env:
        assert env.is_main is True
        barrier(env)


def test_all_reduce_mean_single_process_identity(monkeypatch) -> None:
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)
    env = setup_distributed()
    t = torch.tensor(1.5)
    out = all_reduce_mean(t, env)
    assert torch.equal(out, t)
    teardown_distributed(env)
