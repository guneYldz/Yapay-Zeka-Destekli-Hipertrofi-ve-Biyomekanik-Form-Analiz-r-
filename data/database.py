from __future__ import annotations

import os
from datetime import datetime
import bcrypt

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="kullanıcı")
    height: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
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


def register_user(username: str, password: str) -> bool:
    """Yeni bir kullanici kaydeder. Kullanici adi zaten varsa False doner."""
    with SessionLocal() as session:
        existing = session.query(UserModel).filter_by(user_name=username).first()
        if existing:
            return False
        
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        role = "admin" if username.lower() == "admin" else "kullanıcı"
        new_user = UserModel(user_name=username, password_hash=hashed, role=role)
        session.add(new_user)
        session.commit()
        return True


def verify_user(username: str, password: str) -> dict | None:
    """Kullanici adi ve sifreyi dogrular. Basariliysa kullanici objesini sozluk olarak doner, degilse None."""
    with SessionLocal() as session:
        user = session.query(UserModel).filter_by(user_name=username).first()
        if not user:
            return None
        
        # Eger veritabaninda daha onceden plaintext kaydedilmis bir sifre varsa (migration oncesi vs)
        is_valid = False
        if not user.password_hash.startswith("$2b$") and not user.password_hash.startswith("$2a$"):
            if user.password_hash == password:
                is_valid = True
        elif bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            is_valid = True

        if is_valid:
            user.last_login_at = datetime.utcnow()
            session.commit()
            return {
                "id": user.id,
                "username": user.user_name,
                "role": user.role,
                "height": user.height,
                "weight": user.weight
            }
        return None

def update_user_stats(user_id: int, height: float, weight: float) -> bool:
    with SessionLocal() as session:
        user = session.query(UserModel).filter_by(id=user_id).first()
        if user:
            user.height = height
            user.weight = weight
            session.commit()
            return True
        return False

def get_all_users() -> list[dict]:
    with SessionLocal() as session:
        users = session.query(UserModel).all()
        return [{"id": u.id, "username": u.user_name, "role": u.role} for u in users]

def delete_user(user_id: int) -> bool:
    with SessionLocal() as session:
        user = session.query(UserModel).filter_by(id=user_id).first()
        if user:
            session.delete(user)
            session.commit()
            return True
        return False
