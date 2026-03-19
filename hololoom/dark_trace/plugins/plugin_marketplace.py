"""
Dark Trace Plugin Marketplace

Community plugin repository with ratings, reviews, and discovery
for sharing and distributing Dark Trace interpretability plugins.

Author: HoloLoom Team
Created: December 2025
"""

import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from hololoom.dark_trace.plugins.plugin_loader import (
    LoadedPlugin,
    PluginLoader,
)
from hololoom.dark_trace.plugins.plugin_protocol import (
    PluginCategory,
    PluginMetadata,
    PluginRating,
)
from hololoom.dark_trace.plugins.plugin_signing import (
    PluginVerifier,
    SignedManifest,
    create_verifier,
)

logger = logging.getLogger(__name__)


class SortOrder(Enum):
    """Sort order for plugin listings."""
    RATING = "rating"
    DOWNLOADS = "downloads"
    UPDATED = "updated"
    NAME = "name"
    TRENDING = "trending"


@dataclass
class PluginDownloadStats:
    """Download statistics for a plugin."""
    total_downloads: int = 0
    weekly_downloads: int = 0
    monthly_downloads: int = 0


@dataclass
class MarketplacePlugin:
    """A plugin in the marketplace."""
    metadata: PluginMetadata
    signed_manifest: SignedManifest | None = None

    # Statistics
    download_stats: PluginDownloadStats = field(default_factory=PluginDownloadStats)
    average_rating: float = 0.0
    rating_count: int = 0
    ratings: list[PluginRating] = field(default_factory=list)

    # Marketplace metadata
    verified: bool = False
    featured: bool = False
    deprecated: bool = False
    deprecation_message: str | None = None

    # Source
    source_url: str | None = None
    package_hash: str | None = None
    package_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "signed_manifest": self.signed_manifest.to_dict() if self.signed_manifest else None,
            "download_stats": {
                "total": self.download_stats.total_downloads,
                "weekly": self.download_stats.weekly_downloads,
                "monthly": self.download_stats.monthly_downloads,
            },
            "average_rating": self.average_rating,
            "rating_count": self.rating_count,
            "ratings": [r.to_dict() for r in self.ratings[:10]],  # Top 10 reviews
            "verified": self.verified,
            "featured": self.featured,
            "deprecated": self.deprecated,
            "deprecation_message": self.deprecation_message,
            "source_url": self.source_url,
            "package_hash": self.package_hash,
            "package_size": self.package_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketplacePlugin":
        metadata = PluginMetadata.from_dict(data["metadata"])
        signed_manifest = None
        if data.get("signed_manifest"):
            signed_manifest = SignedManifest.from_dict(data["signed_manifest"])

        plugin = cls(
            metadata=metadata,
            signed_manifest=signed_manifest,
            verified=data.get("verified", False),
            featured=data.get("featured", False),
            deprecated=data.get("deprecated", False),
            deprecation_message=data.get("deprecation_message"),
            source_url=data.get("source_url"),
            package_hash=data.get("package_hash"),
            package_size=data.get("package_size", 0),
        )

        stats = data.get("download_stats", {})
        plugin.download_stats = PluginDownloadStats(
            total_downloads=stats.get("total", 0),
            weekly_downloads=stats.get("weekly", 0),
            monthly_downloads=stats.get("monthly", 0),
        )

        plugin.average_rating = data.get("average_rating", 0.0)
        plugin.rating_count = data.get("rating_count", 0)

        return plugin


class PluginMarketplace:
    """
    Community plugin marketplace for Dark Trace.

    Features:
    - Plugin discovery and search
    - Ratings and reviews
    - Download and installation
    - Signature verification
    - Featured and trending plugins
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        cache_path: Path | None = None,
        verifier: PluginVerifier | None = None,
    ):
        self.storage_path = storage_path or Path.home() / ".hololoom" / "marketplace"
        self.cache_path = cache_path or Path.home() / ".hololoom" / "plugin_cache"
        self.verifier = verifier or create_verifier()

        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Plugin registry (in-memory, backed by storage)
        self._plugins: dict[str, MarketplacePlugin] = {}
        self._ratings: dict[str, list[PluginRating]] = {}

        # Load existing data
        self._load_registry()

    def publish(
        self,
        plugin_dir: Path,
        metadata: PluginMetadata,
        signed_manifest: SignedManifest | None = None,
        source_url: str | None = None,
    ) -> MarketplacePlugin:
        """
        Publish a plugin to the marketplace.

        Args:
            plugin_dir: Directory containing plugin files
            metadata: Plugin metadata
            signed_manifest: Optional signed manifest for verification
            source_url: Optional URL to plugin source

        Returns:
            MarketplacePlugin
        """
        plugin_key = f"{metadata.name}:{metadata.version}"

        # Create package
        package_path = self._create_package(plugin_dir, metadata)
        package_hash = self._hash_file(package_path)
        package_size = package_path.stat().st_size

        # Store package
        dest_path = self.storage_path / "packages" / f"{metadata.name}-{metadata.version}.zip"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(package_path, dest_path)

        # Verify if signed
        verified = False
        if signed_manifest:
            valid, msg = self.verifier.full_verify(plugin_dir, signed_manifest)
            verified = valid
            if not valid:
                logger.warning(f"Plugin verification failed: {msg}")

        # Create marketplace entry
        mp_plugin = MarketplacePlugin(
            metadata=metadata,
            signed_manifest=signed_manifest,
            verified=verified,
            source_url=source_url,
            package_hash=package_hash,
            package_size=package_size,
        )

        self._plugins[plugin_key] = mp_plugin
        self._save_registry()

        logger.info(f"Published plugin: {metadata.name} v{metadata.version}")
        return mp_plugin

    def unpublish(self, name: str, version: str) -> bool:
        """
        Remove a plugin from the marketplace.

        Args:
            name: Plugin name
            version: Plugin version

        Returns:
            True if removed successfully
        """
        plugin_key = f"{name}:{version}"
        if plugin_key not in self._plugins:
            return False

        # Remove package file
        package_path = self.storage_path / "packages" / f"{name}-{version}.zip"
        if package_path.exists():
            package_path.unlink()

        del self._plugins[plugin_key]
        self._save_registry()

        logger.info(f"Unpublished plugin: {name} v{version}")
        return True

    def search(
        self,
        query: str | None = None,
        category: PluginCategory | None = None,
        tags: list[str] | None = None,
        verified_only: bool = False,
        sort_by: SortOrder = SortOrder.RATING,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MarketplacePlugin]:
        """
        Search for plugins in the marketplace.

        Args:
            query: Search query (matches name, description, author)
            category: Filter by category
            tags: Filter by tags (any match)
            verified_only: Only show verified plugins
            sort_by: Sort order
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching plugins
        """
        results = []

        for mp_plugin in self._plugins.values():
            # Skip deprecated unless explicitly searched
            if mp_plugin.deprecated:
                continue

            # Category filter
            if category and mp_plugin.metadata.category != category:
                continue

            # Verified filter
            if verified_only and not mp_plugin.verified:
                continue

            # Tag filter
            if tags:
                if not any(tag in mp_plugin.metadata.tags for tag in tags):
                    continue

            # Query filter
            if query:
                query_lower = query.lower()
                if not any([
                    query_lower in mp_plugin.metadata.name.lower(),
                    query_lower in mp_plugin.metadata.description.lower(),
                    query_lower in mp_plugin.metadata.author.lower(),
                    any(query_lower in tag.lower() for tag in mp_plugin.metadata.tags),
                ]):
                    continue

            results.append(mp_plugin)

        # Sort
        if sort_by == SortOrder.RATING:
            results.sort(key=lambda p: p.average_rating, reverse=True)
        elif sort_by == SortOrder.DOWNLOADS:
            results.sort(key=lambda p: p.download_stats.total_downloads, reverse=True)
        elif sort_by == SortOrder.UPDATED:
            results.sort(
                key=lambda p: p.metadata.updated_at or datetime.min,
                reverse=True
            )
        elif sort_by == SortOrder.NAME:
            results.sort(key=lambda p: p.metadata.name.lower())
        elif sort_by == SortOrder.TRENDING:
            # Trending = combination of recent downloads and ratings
            results.sort(
                key=lambda p: (
                    p.download_stats.weekly_downloads * 0.5 +
                    p.average_rating * 20
                ),
                reverse=True
            )

        return results[offset:offset + limit]

    def get_plugin(self, name: str, version: str | None = None) -> MarketplacePlugin | None:
        """
        Get a specific plugin.

        Args:
            name: Plugin name
            version: Specific version (latest if not specified)

        Returns:
            MarketplacePlugin or None
        """
        if version:
            return self._plugins.get(f"{name}:{version}")

        # Find latest version
        candidates = [
            mp for key, mp in self._plugins.items()
            if key.startswith(f"{name}:")
        ]
        if not candidates:
            return None

        # Sort by version (simple string sort, works for semver)
        candidates.sort(key=lambda p: p.metadata.version, reverse=True)
        return candidates[0]

    def get_versions(self, name: str) -> list[str]:
        """Get all available versions of a plugin."""
        versions = []
        for key in self._plugins:
            if key.startswith(f"{name}:"):
                versions.append(key.split(":", 1)[1])
        versions.sort(reverse=True)
        return versions

    def download(
        self,
        name: str,
        version: str | None = None,
        dest_dir: Path | None = None,
    ) -> Path | None:
        """
        Download a plugin package.

        Args:
            name: Plugin name
            version: Specific version (latest if not specified)
            dest_dir: Destination directory

        Returns:
            Path to downloaded package or None
        """
        mp_plugin = self.get_plugin(name, version)
        if not mp_plugin:
            logger.error(f"Plugin not found: {name}")
            return None

        version = mp_plugin.metadata.version
        package_path = self.storage_path / "packages" / f"{name}-{version}.zip"

        if not package_path.exists():
            logger.error(f"Package file not found: {package_path}")
            return None

        # Verify hash
        actual_hash = self._hash_file(package_path)
        if mp_plugin.package_hash and actual_hash != mp_plugin.package_hash:
            logger.error(f"Package hash mismatch for {name}")
            return None

        # Copy to destination
        if dest_dir:
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{name}-{version}.zip"
            shutil.copy(package_path, dest_path)
        else:
            dest_path = package_path

        # Update download stats
        mp_plugin.download_stats.total_downloads += 1
        mp_plugin.download_stats.weekly_downloads += 1
        mp_plugin.download_stats.monthly_downloads += 1
        self._save_registry()

        logger.info(f"Downloaded plugin: {name} v{version}")
        return dest_path

    def install(
        self,
        name: str,
        version: str | None = None,
        install_dir: Path | None = None,
        loader: PluginLoader | None = None,
    ) -> LoadedPlugin | None:
        """
        Download and install a plugin.

        Args:
            name: Plugin name
            version: Specific version (latest if not specified)
            install_dir: Installation directory
            loader: Plugin loader to use

        Returns:
            LoadedPlugin or None
        """
        # Download package
        package_path = self.download(name, version, self.cache_path)
        if not package_path:
            return None

        # Extract to install directory
        install_dir = install_dir or self.cache_path / "installed" / name
        install_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.unpack_archive(package_path, install_dir)
        except Exception as e:
            logger.error(f"Failed to extract plugin: {e}")
            return None

        # Load plugin
        if loader:
            return loader.load_from_directory(install_dir)[0] if loader.load_from_directory(install_dir) else None

        return None

    # Rating and reviews
    def rate(
        self,
        name: str,
        version: str,
        user_id: str,
        rating: float,
        review: str | None = None,
    ) -> PluginRating | None:
        """
        Rate a plugin.

        Args:
            name: Plugin name
            version: Plugin version
            user_id: ID of user rating
            rating: Rating 1-5
            review: Optional review text

        Returns:
            PluginRating or None
        """
        if not 1 <= rating <= 5:
            logger.error("Rating must be between 1 and 5")
            return None

        mp_plugin = self.get_plugin(name, version)
        if not mp_plugin:
            return None

        plugin_key = f"{name}:{version}"

        # Check for existing rating from user
        if plugin_key not in self._ratings:
            self._ratings[plugin_key] = []

        existing = [r for r in self._ratings[plugin_key] if r.user_id == user_id]
        if existing:
            # Update existing rating
            existing[0].rating = rating
            existing[0].review = review
            new_rating = existing[0]
        else:
            # Add new rating
            new_rating = PluginRating(
                plugin_name=name,
                plugin_version=version,
                user_id=user_id,
                rating=rating,
                review=review,
            )
            self._ratings[plugin_key].append(new_rating)
            mp_plugin.rating_count += 1

        # Recalculate average
        all_ratings = self._ratings[plugin_key]
        mp_plugin.average_rating = sum(r.rating for r in all_ratings) / len(all_ratings)
        mp_plugin.ratings = sorted(all_ratings, key=lambda r: r.helpful_votes, reverse=True)

        self._save_registry()
        return new_rating

    def get_ratings(
        self,
        name: str,
        version: str | None = None,
        limit: int = 10,
        sort_by: str = "helpful",  # "helpful", "recent", "rating"
    ) -> list[PluginRating]:
        """
        Get ratings for a plugin.

        Args:
            name: Plugin name
            version: Specific version (all if not specified)
            limit: Maximum ratings to return
            sort_by: Sort order

        Returns:
            List of PluginRating
        """
        all_ratings = []

        for key, ratings in self._ratings.items():
            if version and key != f"{name}:{version}":
                continue
            if not version and not key.startswith(f"{name}:"):
                continue
            all_ratings.extend(ratings)

        # Sort
        if sort_by == "helpful":
            all_ratings.sort(key=lambda r: r.helpful_votes, reverse=True)
        elif sort_by == "recent":
            all_ratings.sort(key=lambda r: r.created_at, reverse=True)
        elif sort_by == "rating":
            all_ratings.sort(key=lambda r: r.rating, reverse=True)

        return all_ratings[:limit]

    def mark_helpful(self, name: str, version: str, user_id: str) -> bool:
        """Mark a review as helpful."""
        plugin_key = f"{name}:{version}"
        if plugin_key not in self._ratings:
            return False

        for rating in self._ratings[plugin_key]:
            if rating.user_id == user_id:
                # Can't mark own review
                continue
            rating.helpful_votes += 1
            self._save_registry()
            return True

        return False

    # Featured and trending
    def get_featured(self, limit: int = 10) -> list[MarketplacePlugin]:
        """Get featured plugins."""
        featured = [p for p in self._plugins.values() if p.featured and not p.deprecated]
        featured.sort(key=lambda p: p.average_rating, reverse=True)
        return featured[:limit]

    def get_trending(self, limit: int = 10) -> list[MarketplacePlugin]:
        """Get trending plugins (high recent activity)."""
        return self.search(sort_by=SortOrder.TRENDING, limit=limit)

    def get_new(self, limit: int = 10) -> list[MarketplacePlugin]:
        """Get newly added plugins."""
        new_plugins = [p for p in self._plugins.values() if not p.deprecated]
        new_plugins.sort(
            key=lambda p: p.metadata.created_at or datetime.min,
            reverse=True
        )
        return new_plugins[:limit]

    def set_featured(self, name: str, version: str, featured: bool = True) -> bool:
        """Mark a plugin as featured."""
        mp_plugin = self.get_plugin(name, version)
        if not mp_plugin:
            return False

        mp_plugin.featured = featured
        self._save_registry()
        return True

    def deprecate(
        self,
        name: str,
        version: str,
        message: str | None = None,
    ) -> bool:
        """Mark a plugin as deprecated."""
        mp_plugin = self.get_plugin(name, version)
        if not mp_plugin:
            return False

        mp_plugin.deprecated = True
        mp_plugin.deprecation_message = message
        self._save_registry()
        return True

    # Category browsing
    def get_categories(self) -> dict[PluginCategory, int]:
        """Get plugin count by category."""
        counts = dict.fromkeys(PluginCategory, 0)
        for mp_plugin in self._plugins.values():
            if not mp_plugin.deprecated:
                counts[mp_plugin.metadata.category] += 1
        return counts

    def get_by_category(
        self,
        category: PluginCategory,
        limit: int = 50,
    ) -> list[MarketplacePlugin]:
        """Get plugins by category."""
        return self.search(category=category, limit=limit)

    # Statistics
    def get_stats(self) -> dict[str, Any]:
        """Get marketplace statistics."""
        total_plugins = len(self._plugins)
        verified_count = sum(1 for p in self._plugins.values() if p.verified)
        total_downloads = sum(
            p.download_stats.total_downloads for p in self._plugins.values()
        )
        total_ratings = sum(p.rating_count for p in self._plugins.values())

        return {
            "total_plugins": total_plugins,
            "verified_plugins": verified_count,
            "total_downloads": total_downloads,
            "total_ratings": total_ratings,
            "categories": {
                cat.value: count
                for cat, count in self.get_categories().items()
            },
        }

    # Private methods
    def _create_package(self, plugin_dir: Path, metadata: PluginMetadata) -> Path:
        """Create a zip package from plugin directory."""
        temp_dir = Path(tempfile.mkdtemp())
        package_path = temp_dir / f"{metadata.name}-{metadata.version}.zip"

        shutil.make_archive(
            str(package_path.with_suffix("")),
            "zip",
            plugin_dir,
        )

        return package_path

    def _hash_file(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_registry(self) -> None:
        """Load plugin registry from storage."""
        registry_path = self.storage_path / "registry.json"
        if not registry_path.exists():
            return

        try:
            with open(registry_path) as f:
                data = json.load(f)

            for key, plugin_data in data.get("plugins", {}).items():
                self._plugins[key] = MarketplacePlugin.from_dict(plugin_data)

            for key, ratings_data in data.get("ratings", {}).items():
                self._ratings[key] = [
                    PluginRating(
                        plugin_name=r["plugin_name"],
                        plugin_version=r["plugin_version"],
                        user_id=r["user_id"],
                        rating=r["rating"],
                        review=r.get("review"),
                        created_at=datetime.fromisoformat(r["created_at"]),
                        helpful_votes=r.get("helpful_votes", 0),
                    )
                    for r in ratings_data
                ]

            logger.info(f"Loaded {len(self._plugins)} plugins from registry")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save plugin registry to storage."""
        registry_path = self.storage_path / "registry.json"

        data = {
            "plugins": {
                key: plugin.to_dict()
                for key, plugin in self._plugins.items()
            },
            "ratings": {
                key: [r.to_dict() for r in ratings]
                for key, ratings in self._ratings.items()
            },
            "updated_at": datetime.now().isoformat(),
        }

        try:
            with open(registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")


# Factory functions
def create_marketplace(
    storage_path: Path | None = None,
    cache_path: Path | None = None,
) -> PluginMarketplace:
    """Create a plugin marketplace."""
    return PluginMarketplace(storage_path, cache_path)


def get_default_marketplace() -> PluginMarketplace:
    """Get the default marketplace instance."""
    return create_marketplace()
