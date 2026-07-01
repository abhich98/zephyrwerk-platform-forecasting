from collections.abc import Generator
from sqlalchemy.orm import sessionmaker
from db.database import SessionLocal

def get_db() -> Generator:
    """
    Dependency function to provide a database session.

    Yields:
        Generator: A generator that yields a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()