from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./questions.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def add_column_if_missing(target_engine, table: str, column: str, ddl: str):
    """SQLite 不支持 IF NOT EXISTS 加列，用异常兜底做幂等迁移"""
    with target_engine.begin() as conn:
        try:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        except OperationalError:
            pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
