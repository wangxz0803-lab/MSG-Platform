"""Distributed training helpers: init, teardown, rank/world bookkeeping."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    """Snapshot of the distributed runtime at training entry."""

    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device
    initialized: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.initialized and self.world_size > 1


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def setup_distributed(prefer_backend: str | None = None) -> DistEnv:
    """Initialize the process group when ``WORLD_SIZE`` is set."""
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    use_cuda = torch.cuda.is_available()
    backend = prefer_backend or ("nccl" if use_cuda else "gloo")

    needs_pg = "WORLD_SIZE" in os.environ and world_size >= 1 and "MASTER_ADDR" in os.environ
    initialized = False
    if needs_pg and not dist.is_available():
        needs_pg = False

    if needs_pg and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://",
                                world_size=world_size, rank=rank)
        initialized = True

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return DistEnv(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend if initialized else "none",
        device=device,
        initialized=initialized,
    )


def teardown_distributed(env: DistEnv) -> None:
    """Destroy the process group if we created one."""
    if env.initialized and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def distributed_context(prefer_backend: str | None = None) -> Iterator[DistEnv]:
    """Context manager wrapping setup/teardown."""
    env = setup_distributed(prefer_backend=prefer_backend)
    try:
        yield env
    finally:
        teardown_distributed(env)


def barrier(env: DistEnv) -> None:
    """Rank-synchronisation barrier; no-op in single-process mode."""
    if env.initialized and dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor, env: DistEnv) -> torch.Tensor:
    """Average a scalar tensor across ranks."""
    if not env.is_distributed:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= env.world_size
    return tensor


__all__ = [
    "DistEnv",
    "setup_distributed",
    "teardown_distributed",
    "distributed_context",
    "barrier",
    "all_reduce_mean",
]
