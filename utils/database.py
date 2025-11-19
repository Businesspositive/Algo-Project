from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class FyersToken(Base):
    __tablename__ = "fyers_tokens"

    id = Column(String, primary_key=True)
    access_token = Column(String, nullable=False)
    refresh_token = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# SQLite engine
engine = create_engine("sqlite:///fyers_tokens.db", echo=False)
SessionLocal = sessionmaker(bind=engine)

# Create tables
Base.metadata.create_all(engine)

def save_tokens(access_token: str, refresh_token: str):
    session = SessionLocal()
    token_entry = FyersToken(
        id=str(uuid.uuid4()),
        access_token=access_token,
        refresh_token=refresh_token
    )
    session.add(token_entry)
    session.commit()
    session.close()

def get_latest_tokens():
    session = SessionLocal()
    token_entry = session.query(FyersToken).order_by(FyersToken.created_at.desc()).first()
    session.close()
    if token_entry:
        return token_entry.access_token, token_entry.refresh_token
    return None, None
