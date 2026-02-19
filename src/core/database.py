from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from src.core.config import Settings
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class Database:
    """Database connection manager."""

    def __init__(self, settings: Settings):
        """
        Initialize database connection.

        Args:
            settings (Settings): Application settings.
        """
        self.engine = create_engine(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            echo=settings.database_echo,
            pool_pre_ping=True,  # Verify connections before using
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info("Database connection initialized")

    def create_tables(self):
        """Create all tables (for development only, use Alembic for production)."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            Session: Database session.
        """
        return self.SessionLocal()


# Global database instance
_db_instance: Database = None


def init_database(settings: Settings) -> Database:
    """
    Initialize the global database instance.

    Args:
        settings (Settings): Application settings.

    Returns:
        Database: Database instance.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(settings)
    return _db_instance


def get_db() -> Session:
    """
    FastAPI dependency to get database session.

    Yields:
        Session: Database session that will be closed after request.
    """
    db = _db_instance.get_session()
    try:
        yield db
    finally:
        db.close()
