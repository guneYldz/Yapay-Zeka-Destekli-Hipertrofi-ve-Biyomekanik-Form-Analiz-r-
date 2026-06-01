from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    sessions: Mapped[list["WorkoutSessionModel"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class WorkoutSessionModel(Base):
    __tablename__ = "workout_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    exercise_type: Mapped[str] = mapped_column(String(50), nullable=False, default="Squat")
    total_reps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    user: Mapped[UserModel] = relationship(back_populates="sessions")
    results: Mapped[list["AnalysisResultModel"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class AnalysisResultModel(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("workout_sessions.id"), nullable=False)
    rep_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_knee_angle: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_type: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    risk_level: Mapped[str] = mapped_column(String(30), nullable=False, default="Dusuk")
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    session: Mapped[WorkoutSessionModel] = relationship(back_populates="results")


def _db_path() -> str:
    # Veritabani dosyasi proje ana dizininde tutulur.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "squat_analyzer.db")


engine = create_engine(f"sqlite:///{_db_path()}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def init_db() -> None:
    # Tablolari otomatik olusturur.
    Base.metadata.create_all(bind=engine)
