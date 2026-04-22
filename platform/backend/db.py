"""SQLAlchemy engine, session and declarative Base for the backend."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .settings import get_settings


class Base(DeclarativeBase):
    """Declarative base for all backend ORM models."""


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _connect_args(db_url: str) -> dict[str, Any]:
    """Return driver-specific connect_args for the given URL."""
    if db_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def get_engine() -> Engine:
    """Return the cached engine, creating it on first call."""
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().db_url
        _engine = create_engine(url, connect_args=_connect_args(url), future=True)
        _SessionLocal = sessionmaker(
            bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Return the cached session factory."""
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def reset_engine() -> None:
    """Drop the cached engine/session (used by tests when swapping DBs)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def init_db() -> None:
    """Create all tables declared on :class:`Base.metadata`."""
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    session_local = get_sessionmaker()
    session = session_local()
    try:
        yield session
    finally:
        session.close()
