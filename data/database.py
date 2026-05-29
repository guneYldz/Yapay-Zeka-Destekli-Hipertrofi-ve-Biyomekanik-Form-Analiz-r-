from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def _db_path() -> str:
    # Veritabani dosyasi proje ana dizininde tutulur.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "squat_analyzer.db")


engine = create_engine(f"sqlite:///{_db_path()}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def init_db() -> None:
    # Tablolari otomatik olusturur.
    Base.metadata.create_all(bind=engine)