"""FastAPI application factory + startup hooks for the MSG-Embedding backend."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from msg_embedding.core.logging import get_logger

from . import __version__
from .db import get_sessionmaker, init_db
from .routes import channels, datasets, health, jobs, models, runs, topology
from .routes import configs as configs_route
from .services.artifact_scan import scan_artifacts_and_reports
from .services.manifest_sync import sync_manifest_to_db
from .settings import get_settings

_log = get_logger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

_worker_thread: threading.Thread | None = None
_watcher_thread: threading.Thread | None = None


def _start_embedded_worker() -> None:
    """Start Dramatiq worker + queue watcher as daemon threads."""
    global _worker_thread, _watcher_thread

    try:
        import dramatiq
        from dramatiq.brokers.redis import RedisBroker

        settings = get_settings()
        broker = RedisBroker(url=settings.redis_url)
        dramatiq.set_broker(broker)

        from platform.worker import actors  # noqa: F401

        worker = dramatiq.Worker(broker, worker_threads=4, worker_timeout=1000)
        worker.start()
        _log.info("embedded_worker_started", threads=4)

        from platform.worker.queue_watcher import watch_forever as _wf
        from platform.worker.settings import get_settings as get_worker_settings

        ws = get_worker_settings()
        _watcher_thread = threading.Thread(
            target=_wf, kwargs={"settings": ws}, daemon=True, name="queue-watcher"
        )
        _watcher_thread.start()
        _log.info("embedded_watcher_started")

    except Exception as exc:
        _log.warning("embedded_worker_start_failed", error=str(exc))


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise DB, sync manifest + artifacts, start worker on startup."""
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    session_local = get_sessionmaker()
    with session_local() as session:
        try:
            n_samples = sync_manifest_to_db(session)
        except Exception as exc:
            _log.warning("manifest_sync_failed", error=str(exc))
            n_samples = 0
        try:
            n_runs, n_art = scan_artifacts_and_reports(session)
        except Exception as exc:
            _log.warning("artifact_scan_failed", error=str(exc))
            n_runs = n_art = 0
    _log.info("backend_startup_complete", samples=n_samples, runs=n_runs, artifacts=n_art)
    _start_embedded_worker()
    yield


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="MSG-Embedding Research Platform API",
        description="Backend API for data, jobs, runs, models and configs.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(datasets.router)
    app.include_router(jobs.router)
    app.include_router(runs.router)
    app.include_router(models.router)
    app.include_router(configs_route.router)
    app.include_router(topology.router)
    app.include_router(channels.router)

    if FRONTEND_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="static")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file = FRONTEND_DIST / full_path
            if file.is_file():
                return FileResponse(file)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
