"""Abstract base class for all ChannelSample data sources.

Every concrete source (QuaDRiGa legacy, QuaDRiGa multi-cell, Sionna RT,
internal simulator, field data) subclasses :class:`DataSource` and is
registered via :func:`msg_embedding.data.sources.register_source` so the
pipeline can look them up by name.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any, ClassVar

try:  # OmegaConf is the canonical config carrier but is optional at import time.
    from omegaconf import DictConfig  # type: ignore
except ImportError:  # pragma: no cover - fall back to a typing alias
    DictConfig = Any  # type: ignore[misc,assignment]

from ..contract import ChannelSample

# ---------------------------------------------------------------------------
# Registry (lives here so stub modules can decorate without circular imports)
# ---------------------------------------------------------------------------

SOURCE_REGISTRY: dict[str, type[DataSource]] = {}


def register_source(cls: type[DataSource]) -> type[DataSource]:
    """Class decorator: add ``cls`` to :data:`SOURCE_REGISTRY` keyed by ``cls.name``."""
    if not getattr(cls, "name", ""):
        raise ValueError(
            f"{cls.__name__} must set a non-empty class attribute `name` before registering"
        )
    if cls.name in SOURCE_REGISTRY and SOURCE_REGISTRY[cls.name] is not cls:
        raise ValueError(f"DataSource name conflict: {cls.name!r} already registered")
    SOURCE_REGISTRY[cls.name] = cls
    return cls


class DataSource(abc.ABC):
    """Abstract producer of :class:`ChannelSample` records.

    Subclasses set the class-level ``name`` attribute and implement
    :meth:`iter_samples` and :meth:`describe`. Concrete sources should also
    override :meth:`validate_config` to fail fast on malformed configs.
    """

    #: Unique lookup key used by the registry (e.g. ``"quadriga_real"``).
    name: ClassVar[str] = ""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.validate_config()

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def iter_samples(self) -> Iterator[ChannelSample]:
        """Yield :class:`ChannelSample` instances one at a time."""
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return metadata describing the source.

        Expected keys (minimum): ``expected_sample_count``, ``dimensions``
        (a dict with T / RB / BS_ant / UE_ant / K), and ``scenario`` (a free
        form string identifying the simulation setup).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------
    def validate_config(self) -> None:
        """Source-specific config validation. Override in subclasses."""
        return None


__all__ = ["DataSource", "SOURCE_REGISTRY", "register_source"]
