"""
Plugin and PluginInstance models
"""
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
import uuid

from core.database import Base


class Plugin(Base):
    """Plugin model"""

    __tablename__ = "plugins"

    # Primary Key
    plugin_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic Info
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    version = Column(String(20), nullable=False)
    author = Column(String(100))

    # Type
    plugin_type = Column(
        String(50),
        nullable=False,
        CheckConstraint(
            """plugin_type IN (
                'content', 'integration', 'moderation', 'engagement',
                'communication', 'analytics', 'customization', 'admin'
            )"""
        ),
    )

    # Files
    main_file = Column(String(255), nullable=False)
    config_schema = Column(JSONB, default={})

    # Permissions
    required_permissions = Column(ARRAY(String), default=[])
    hooks = Column(ARRAY(String), default=[])

    # Status
    status = Column(
        String(20),
        default="active",
        CheckConstraint("status IN ('active', 'disabled', 'deprecated')"),
    )
    is_official = Column(Boolean, default=False)

    # Marketplace
    downloads = Column(Integer, default=0)
    rating = Column(Integer, default=0)
    review_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    published_at = Column(TIMESTAMP)

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<Plugin {self.name} v{self.version}>"


class PluginInstance(Base):
    """Plugin instance model (installed plugins)"""

    __tablename__ = "plugin_instances"

    # Primary Key
    instance_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    plugin_id = Column(
        UUID(as_uuid=True),
        ForeignKey("plugins.plugin_id", ondelete="CASCADE"),
        nullable=False,
    )
    community_id = Column(
        UUID(as_uuid=True),
        ForeignKey("communities.community_id", ondelete="CASCADE"),
        nullable=True,  # NULL means global/site-wide
    )

    # Configuration
    config = Column(JSONB, default={})
    enabled = Column(Boolean, default=True)

    # Timestamps
    installed_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<PluginInstance plugin={self.plugin_id} community={self.community_id}>"
