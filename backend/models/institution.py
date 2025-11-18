"""Institution model"""
from datetime import datetime
from sqlalchemy import Column, String, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Institution(Base):
    """Educational institution"""
    __tablename__ = "institutions"

    institution_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    domain = Column(String(255), nullable=False, unique=True, index=True)
    settings = Column(JSONB, default=dict)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    users = relationship("User", back_populates="institution")
    courses = relationship("Course", back_populates="institution")
    plugin_configurations = relationship("PluginConfiguration", back_populates="institution")

    def __repr__(self):
        return f"<Institution {self.name} ({self.domain})>"
