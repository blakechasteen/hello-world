"""Plugin models"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Boolean, TIMESTAMP, ForeignKey, CheckConstraint, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Plugin(Base):
    """Installed plugin"""
    __tablename__ = "plugins"

    plugin_id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    category = Column(String(50), nullable=False, index=True)
    description = Column(Text)
    author = Column(String(255))
    permissions = Column(ARRAY(Text))
    hooks = Column(ARRAY(Text))
    api_routes = Column(JSONB)
    ui_components = Column(JSONB)
    config_schema = Column(JSONB)
    status = Column(String(50), default="inactive", index=True)
    installed_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("status IN ('active', 'inactive', 'deprecated')", name="plugins_status_check"),
        UniqueConstraint('plugin_id', 'version', name='unique_plugin_version'),
    )

    # Relationships
    configurations = relationship("PluginConfiguration", back_populates="plugin")

    def __repr__(self):
        return f"<Plugin {self.plugin_id} v{self.version} ({self.status})>"


class PluginConfiguration(Base):
    """Institution-specific plugin configuration"""
    __tablename__ = "plugin_configurations"

    config_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    institution_id = Column(UUID(as_uuid=True), ForeignKey("institutions.institution_id", ondelete="CASCADE"), nullable=False, index=True)
    plugin_id = Column(String(100), ForeignKey("plugins.plugin_id", ondelete="CASCADE"), nullable=False, index=True)
    config = Column(JSONB, nullable=False, default=dict)
    enabled = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('institution_id', 'plugin_id', name='unique_plugin_config'),
    )

    # Relationships
    institution = relationship("Institution", back_populates="plugin_configurations")
    plugin = relationship("Plugin", back_populates="configurations")

    def __repr__(self):
        return f"<PluginConfiguration {self.plugin_id} for {self.institution_id}>"
