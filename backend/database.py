from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'data', 'feedback.db')

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
Base = declarative_base()

class FeedbackResult(Base):
    __tablename__ = 'feedback_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_text = Column(String, nullable=False)
    cleaned_text = Column(String)
    sentiment = Column(String)
    sentiment_confidence = Column(Float)
    intent = Column(String)
    intent_confidence = Column(Float)
    needs_review = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)

def save_feedback_result(result: dict):
    session = SessionLocal()
    try:
        entry = FeedbackResult(
            original_text=result["original_text"],
            cleaned_text=result["cleaned_text"],
            sentiment=result["sentiment"],
            sentiment_confidence=result["sentiment_confidence"],
            intent=result["intent"],
            intent_confidence=result["intent_confidence"],
            needs_review=result["needs_review"]
        )
        session.add(entry)
        session.commit()
    finally:
        session.close()

def get_all_feedback():
    session = SessionLocal()
    try:
        return session.query(FeedbackResult).order_by(FeedbackResult.created_at.desc()).all()
    finally:
        session.close()

def delete_feedback(entry_id: int):
    """Delete a single feedback entry by its ID."""
    session = SessionLocal()
    try:
        entry = session.query(FeedbackResult).filter(FeedbackResult.id == entry_id).first()
        if entry:
            session.delete(entry)
            session.commit()
            return True
        return False
    finally:
        session.close()

def delete_all_feedback():
    """Delete every feedback entry (use with caution)."""
    session = SessionLocal()
    try:
        session.query(FeedbackResult).delete()
        session.commit()
    finally:
        session.close()