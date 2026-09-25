"""Database engine, session factory and schema creation."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

from sqlalchemy import create_engine, event
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


if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        """Turn on SQLite foreign-key enforcement for every connection.

        SQLite ignores ``ON DELETE CASCADE`` and other referential actions
        unless this pragma is set, and it is off by default. Without it the
        parent/child relationships in this schema are advisory only: deleting a
        meeting would silently leave its transcript, minutes and export rows
        behind. Those orphans are not just wasted space; if SQLite later reuses
        the freed primary key, a new meeting can inherit the stale rows.
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

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
    from app.models import meeting, minutes, transcript  # noqa: F401

    Base.metadata.create_all(bind=engine)