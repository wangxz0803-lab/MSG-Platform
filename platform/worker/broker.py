"""Dramatiq broker setup.

Production uses ``RedisBroker``. Tests use ``StubBroker`` via
:func:`install_stub_broker` -- this must be called *before* importing
``platform.worker.actors`` so the actor decorators bind to the stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .settings import get_settings

if TYPE_CHECKING:
    import dramatiq


def make_redis_broker() -> dramatiq.Broker:
    """Build and register a RedisBroker against ``settings.redis_url``."""
    import dramatiq
    from dramatiq.brokers.redis import RedisBroker

    broker = RedisBroker(url=get_settings().redis_url)
    dramatiq.set_broker(broker)
    return broker


def install_stub_broker() -> dramatiq.Broker:
    """Replace the global broker with a :class:`StubBroker` (for tests)."""
    import dramatiq
    from dramatiq.brokers.stub import StubBroker

    broker = StubBroker()
    broker.emit_after("process_boot")
    dramatiq.set_broker(broker)
    return broker


def current_broker() -> dramatiq.Broker:
    """Return the currently registered Dramatiq broker."""
    import dramatiq

    return dramatiq.get_broker()
