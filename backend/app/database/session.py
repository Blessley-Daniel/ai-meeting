"""Database engine, session factory and schema creation."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# SQLite forbids sharing a connection across threads by default. FastAPI runs
# sync endpoints in a threadpool, so we relax that check; each request still
# gets its own session.
_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def utcnow() -> datetime:
    """Current UTC time as a naive datetime.

    SQLite does not persist timezone offsets, so storing aware datetimes would
    silently come back naive on read. Normalising on write keeps writes and
    reads consistent.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create any missing tables.

    Sufficient for a single-file SQLite database. A larger deployment would use
    Alembic migrations instead.
    """
    # Importing the models registers them on Base.metadata.
    from app.database.base import Base
    from app.models import meeting, transcript  # noqa: F401

    Base.metadata.create_all(bind=engine)