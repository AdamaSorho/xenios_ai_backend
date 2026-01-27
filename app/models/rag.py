"""SQLAlchemy models for RAG system (Spec 0004)."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase

# pgvector support
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for environments without pgvector
    Vector = None


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class Embedding(Base):
    """Vector embeddings for client health data."""

    __tablename__ = "embeddings"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Source reference
    source_type = Column(String(50), nullable=False)
    source_id = Column(Text, nullable=False)
    source_table = Column(String(100))

    # Content
    content_text = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)

    # Embedding vector (1536 dimensions for ada-002)
    # Using raw Column type to avoid import issues; actual type is vector(1536)
    embedding = Column(Vector(1536) if Vector else Text, nullable=False)

    # Metadata for filtering
    metadata_ = Column("metadata", JSONB, default=dict)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default="now()")
    updated_at = Column(DateTime(timezone=True), server_default="now()", onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, client_id={self.client_id}, source_type={self.source_type})>"


class ChatHistory(Base):
    """Chat conversation history for RAG context continuity."""

    __tablename__ = "chat_history"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    client_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    coach_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Message content
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)

    # RAG metadata
    sources_used = Column(JSONB)  # List of source citations
    tokens_used = Column(Integer)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default="now()")

    def __repr__(self) -> str:
        return f"<ChatHistory(id={self.id}, conversation_id={self.conversation_id}, role={self.role})>"


class InsightGenerationLog(Base):
    """Log of insight generation attempts for deduplication and analytics."""

    __tablename__ = "insight_generation_log"
    __table_args__ = {"schema": "ai_backend"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    coach_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Trigger info
    trigger = Column(String(50), nullable=False)
    triggering_data = Column(JSONB, nullable=False)

    # Result
    insight_id = Column(UUID(as_uuid=True))  # FK to public.insights if generated
    insight_type = Column(String(50))
    title = Column(Text)
    title_embedding = Column(Vector(1536) if Vector else Text)  # For deduplication
    status = Column(String(20), nullable=False)  # generated, duplicate, failed
    error_message = Column(Text)

    # Metrics
    context_items_used = Column(Integer)
    tokens_used = Column(Integer)
    generation_time_ms = Column(Integer)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default="now()")

    def __repr__(self) -> str:
        return f"<InsightGenerationLog(id={self.id}, client_id={self.client_id}, status={self.status})>"
