"""
Async MySQL database connection using SQLAlchemy.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from app.config import get_settings

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            echo=settings.environment == "development",
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncSession: #type:ignore
    """FastAPI dependency to get a DB session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session #type: ignore


async def startup_db():
    """Test DB connection on startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))


async def shutdown_db():
    """Close DB pool on shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
