#!/usr/bin/env python3
"""
Workflow Marketplace

Platform for sharing and discovering reusable workflow templates:
- Browse public workflow templates
- Share your own workflows
- Rate and review workflows
- Install workflows with dependencies
- Version control and updates
- Community contributions

Usage:
    from HoloLoom.chatops.workflow_marketplace import WorkflowMarketplace

    marketplace = WorkflowMarketplace()

    # Browse workflows
    workflows = marketplace.search("incident response")

    # Install workflow
    await marketplace.install("incident-investigation-v2", author="devops-team")

    # Publish your workflow
    await marketplace.publish(workflow_def, metadata)

    # Rate workflow
    marketplace.rate("incident-investigation-v2", rating=5, review="Excellent!")
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import hashlib


@dataclass
class WorkflowMetadata:
    """Workflow metadata"""
    workflow_id: str
    name: str
    version: str
    author: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)

    # Requirements
    dependencies: List[str] = field(default_factory=list)
    min_hololoom_version: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)

    # Usage stats
    downloads: int = 0
    rating: float = 0.0
    review_count: int = 0
    usage_count: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    published_at: Optional[datetime] = None

    # Verification
    verified: bool = False
    verified_by: Optional[str] = None


@dataclass
class WorkflowTemplate:
    """Complete workflow template"""
    metadata: WorkflowMetadata
    dsl: str  # Workflow DSL definition
    examples: List[Dict[str, Any]] = field(default_factory=list)
    documentation: str = ""

    # Files
    files: Dict[str, str] = field(default_factory=dict)  # filename -> content

    # Checksum for integrity
    checksum: str = ""


@dataclass
class WorkflowReview:
    """User review of workflow"""
    review_id: str
    workflow_id: str
    author: str
    rating: int  # 1-5
    title: str
    comment: str
    helpful_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowInstallation:
    """Installed workflow record"""
    workflow_id: str
    version: str
    installed_at: datetime
    installed_by: str
    auto_update: bool = False
    customizations: Dict[str, Any] = field(default_factory=dict)


class WorkflowMarketplace:
    """
    Marketplace for sharing and discovering workflows.

    Features:
    - Search and browse workflows
    - Install with dependency resolution
    - Rate and review
    - Version management
    - Community contributions
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_remote: bool = False,
        remote_url: Optional[str] = None
    ):
        self.storage_path = storage_path or Path("./chatops_data/marketplace")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.enable_remote = enable_remote
        self.remote_url = remote_url

        # Local storage
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.reviews: Dict[str, List[WorkflowReview]] = defaultdict(list)
        self.installations: Dict[str, WorkflowInstallation] = {}

        # Categories
        self.categories = [
            "incident-response",
            "deployment",
            "code-review",
            "monitoring",
            "security",
            "onboarding",
            "maintenance",
            "testing",
            "documentation"
        ]

        # Statistics
        self.stats = {
            "total_workflows": 0,
            "total_downloads": 0,
            "total_reviews": 0,
            "avg_rating": 0.0
        }

        # Load data
        self._load_marketplace()

        logging.info("WorkflowMarketplace initialized")

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_rating: float = 0.0,
        verified_only: bool = False
    ) -> List[WorkflowTemplate]:
        """
        Search for workflows.

        Args:
            query: Search query (matches name, description, tags)
            category: Filter by category
            tags: Filter by tags (any match)
            min_rating: Minimum rating
            verified_only: Only verified workflows

        Returns:
            Matching workflow templates
        """

        results = list(self.templates.values())

        # Filter by query
        if query:
            query_lower = query.lower()
            results = [
                w for w in results
                if (query_lower in w.metadata.name.lower() or
                    query_lower in w.metadata.description.lower() or
                    any(query_lower in tag.lower() for tag in w.metadata.tags))
            ]

        # Filter by category
        if category:
            results = [w for w in results if w.metadata.category == category]

        # Filter by tags
        if tags:
            results = [
                w for w in results
                if any(tag in w.metadata.tags for tag in tags)
            ]

        # Filter by rating
        if min_rating > 0:
            results = [w for w in results if w.metadata.rating >= min_rating]

        # Filter verified
        if verified_only:
            results = [w for w in results if w.metadata.verified]

        # Sort by rating, then downloads
        results.sort(
            key=lambda w: (w.metadata.rating, w.metadata.downloads),
            reverse=True
        )

        return results

    async def install(
        self,
        workflow_id: str,
        version: Optional[str] = None,
        auto_update: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Install a workflow.

        Args:
            workflow_id: Workflow to install
            version: Specific version (or latest)
            auto_update: Enable auto-updates
            user_id: Installing user

        Returns:
            Installation result
        """

        # Find workflow
        template = self.templates.get(workflow_id)
        if not template:
            if self.enable_remote:
                # Try to fetch from remote
                template = await self._fetch_remote_workflow(workflow_id, version)
                if not template:
                    raise ValueError(f"Workflow '{workflow_id}' not found")
            else:
                raise ValueError(f"Workflow '{workflow_id}' not found")

        # Use specified version or latest
        install_version = version or template.metadata.version

        # Check dependencies
        dependencies = await self._resolve_dependencies(template)

        # Verify checksum
        if template.checksum:
            calculated = self._calculate_checksum(template)
            if calculated != template.checksum:
                raise ValueError("Checksum verification failed")

        # Install
        installation = WorkflowInstallation(
            workflow_id=workflow_id,
            version=install_version,
            installed_at=datetime.now(),
            installed_by=user_id or "system",
            auto_update=auto_update
        )

        self.installations[workflow_id] = installation

        # Update stats
        template.metadata.downloads += 1
        self.stats["total_downloads"] += 1

        # Save
        self._save_marketplace()

        return {
            "workflow_id": workflow_id,
            "version": install_version,
            "dependencies_installed": dependencies,
            "success": True
        }

    async def publish(
        self,
        dsl: str,
        metadata: Dict[str, Any],
        examples: Optional[List[Dict]] = None,
        documentation: str = "",
        files: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Publish a workflow to marketplace.

        Args:
            dsl: Workflow DSL definition
            metadata: Workflow metadata
            examples: Usage examples
            documentation: Documentation markdown
            files: Additional files

        Returns:
            Workflow ID
        """

        # Create metadata
        workflow_meta = WorkflowMetadata(
            workflow_id=metadata["workflow_id"],
            name=metadata["name"],
            version=metadata.get("version", "1.0.0"),
            author=metadata["author"],
            description=metadata["description"],
            category=metadata.get("category", "general"),
            tags=metadata.get("tags", []),
            dependencies=metadata.get("dependencies", []),
            min_hololoom_version=metadata.get("min_hololoom_version"),
            required_skills=metadata.get("required_skills", []),
            published_at=datetime.now()
        )

        # Create template
        template = WorkflowTemplate(
            metadata=workflow_meta,
            dsl=dsl,
            examples=examples or [],
            documentation=documentation,
            files=files or {}
        )

        # Calculate checksum
        template.checksum = self._calculate_checksum(template)

        # Store
        self.templates[template.metadata.workflow_id] = template
        self.stats["total_workflows"] += 1

        # Save
        self._save_marketplace()

        logging.info(f"Published workflow: {template.metadata.workflow_id} v{template.metadata.version}")

        return template.metadata.workflow_id

    def rate(
        self,
        workflow_id: str,
        rating: int,
        title: str = "",
        comment: str = "",
        author: str = "anonymous"
    ) -> str:
        """
        Rate and review a workflow.

        Args:
            workflow_id: Workflow to rate
            rating: Rating (1-5)
            title: Review title
            comment: Review text
            author: Review author

        Returns:
            Review ID
        """

        if rating < 1 or rating > 5:
            raise ValueError("Rating must be 1-5")

        template = self.templates.get(workflow_id)
        if not template:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        # Create review
        review = WorkflowReview(
            review_id=f"review_{workflow_id}_{len(self.reviews[workflow_id])}",
            workflow_id=workflow_id,
            author=author,
            rating=rating,
            title=title,
            comment=comment
        )

        # Store
        self.reviews[workflow_id].append(review)

        # Update workflow rating
        all_ratings = [r.rating for r in self.reviews[workflow_id]]
        template.metadata.rating = sum(all_ratings) / len(all_ratings)
        template.metadata.review_count = len(all_ratings)

        # Update stats
        self.stats["total_reviews"] += 1
        all_workflow_ratings = [w.metadata.rating for w in self.templates.values() if w.metadata.rating > 0]
        self.stats["avg_rating"] = sum(all_workflow_ratings) / len(all_workflow_ratings) if all_workflow_ratings else 0

        # Save
        self._save_marketplace()

        return review.review_id

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowTemplate]:
        """Get workflow template by ID"""
        return self.templates.get(workflow_id)

    def get_reviews(self, workflow_id: str) -> List[WorkflowReview]:
        """Get reviews for workflow"""
        return self.reviews.get(workflow_id, [])

    def list_installed(self) -> List[WorkflowInstallation]:
        """List installed workflows"""
        return list(self.installations.values())

    async def uninstall(self, workflow_id: str) -> bool:
        """Uninstall a workflow"""
        if workflow_id in self.installations:
            del self.installations[workflow_id]
            self._save_marketplace()
            return True
        return False

    async def update(
        self,
        workflow_id: str,
        to_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an installed workflow.

        Args:
            workflow_id: Workflow to update
            to_version: Target version (or latest)

        Returns:
            Update result
        """

        installation = self.installations.get(workflow_id)
        if not installation:
            raise ValueError(f"Workflow '{workflow_id}' not installed")

        # Check for updates
        template = self.templates.get(workflow_id)
        if not template:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        current_version = installation.version
        latest_version = to_version or template.metadata.version

        if current_version == latest_version:
            return {
                "workflow_id": workflow_id,
                "already_latest": True,
                "version": current_version
            }

        # Perform update (re-install)
        result = await self.install(
            workflow_id,
            version=latest_version,
            auto_update=installation.auto_update,
            user_id=installation.installed_by
        )

        return {
            "workflow_id": workflow_id,
            "previous_version": current_version,
            "new_version": latest_version,
            "success": result["success"]
        }

    async def check_updates(self) -> List[Dict[str, Any]]:
        """Check for workflow updates"""
        updates = []

        for installation in self.installations.values():
            template = self.templates.get(installation.workflow_id)
            if not template:
                continue

            if template.metadata.version != installation.version:
                updates.append({
                    "workflow_id": installation.workflow_id,
                    "current_version": installation.version,
                    "latest_version": template.metadata.version,
                    "auto_update": installation.auto_update
                })

        return updates

    def verify_workflow(self, workflow_id: str, verifier: str):
        """Mark workflow as verified"""
        template = self.templates.get(workflow_id)
        if template:
            template.metadata.verified = True
            template.metadata.verified_by = verifier
            self._save_marketplace()

    def export_workflow(self, workflow_id: str, output_path: Path):
        """Export workflow as JSON bundle"""
        template = self.templates.get(workflow_id)
        if not template:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        bundle = {
            "metadata": {
                "workflow_id": template.metadata.workflow_id,
                "name": template.metadata.name,
                "version": template.metadata.version,
                "author": template.metadata.author,
                "description": template.metadata.description,
                "category": template.metadata.category,
                "tags": template.metadata.tags,
                "dependencies": template.metadata.dependencies
            },
            "dsl": template.dsl,
            "examples": template.examples,
            "documentation": template.documentation,
            "files": template.files,
            "checksum": template.checksum
        }

        with open(output_path, 'w') as f:
            json.dump(bundle, f, indent=2)

        logging.info(f"Exported workflow '{workflow_id}' to {output_path}")

    def import_workflow(self, bundle_path: Path) -> str:
        """Import workflow from JSON bundle"""
        with open(bundle_path, 'r') as f:
            bundle = json.load(f)

        # Verify checksum
        # ... (implementation)

        # Create template
        metadata = WorkflowMetadata(**bundle["metadata"])
        template = WorkflowTemplate(
            metadata=metadata,
            dsl=bundle["dsl"],
            examples=bundle.get("examples", []),
            documentation=bundle.get("documentation", ""),
            files=bundle.get("files", {}),
            checksum=bundle.get("checksum", "")
        )

        # Store
        self.templates[template.metadata.workflow_id] = template
        self._save_marketplace()

        return template.metadata.workflow_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        stats = self.stats.copy()

        # Category breakdown
        category_counts = defaultdict(int)
        for template in self.templates.values():
            category_counts[template.metadata.category] += 1

        stats["by_category"] = dict(category_counts)

        # Top rated
        top_rated = sorted(
            self.templates.values(),
            key=lambda w: w.metadata.rating,
            reverse=True
        )[:5]

        stats["top_rated"] = [
            {
                "workflow_id": w.metadata.workflow_id,
                "name": w.metadata.name,
                "rating": w.metadata.rating
            }
            for w in top_rated
        ]

        # Most downloaded
        most_downloaded = sorted(
            self.templates.values(),
            key=lambda w: w.metadata.downloads,
            reverse=True
        )[:5]

        stats["most_downloaded"] = [
            {
                "workflow_id": w.metadata.workflow_id,
                "name": w.metadata.name,
                "downloads": w.metadata.downloads
            }
            for w in most_downloaded
        ]

        return stats

    async def _resolve_dependencies(self, template: WorkflowTemplate) -> List[str]:
        """Resolve and install dependencies"""
        installed = []

        for dep_id in template.metadata.dependencies:
            if dep_id not in self.installations:
                # Install dependency
                await self.install(dep_id)
                installed.append(dep_id)

        return installed

    def _calculate_checksum(self, template: WorkflowTemplate) -> str:
        """Calculate checksum for template integrity"""
        content = json.dumps({
            "dsl": template.dsl,
            "metadata": {
                "name": template.metadata.name,
                "version": template.metadata.version,
                "author": template.metadata.author
            }
        }, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()

    async def _fetch_remote_workflow(
        self,
        workflow_id: str,
        version: Optional[str]
    ) -> Optional[WorkflowTemplate]:
        """Fetch workflow from remote marketplace"""
        # Stub - would implement HTTP fetch
        return None

    def _save_marketplace(self):
        """Save marketplace data"""

        # Save templates
        templates_file = self.storage_path / "templates.json"
        templates_data = {
            wf_id: {
                "metadata": {
                    "workflow_id": wf.metadata.workflow_id,
                    "name": wf.metadata.name,
                    "version": wf.metadata.version,
                    "author": wf.metadata.author,
                    "description": wf.metadata.description,
                    "category": wf.metadata.category,
                    "tags": wf.metadata.tags,
                    "downloads": wf.metadata.downloads,
                    "rating": wf.metadata.rating,
                    "review_count": wf.metadata.review_count
                },
                "dsl": wf.dsl,
                "checksum": wf.checksum
            }
            for wf_id, wf in self.templates.items()
        }

        with open(templates_file, 'w') as f:
            json.dump(templates_data, f, indent=2)

        # Save reviews
        reviews_file = self.storage_path / "reviews.json"
        reviews_data = {
            wf_id: [
                {
                    "review_id": r.review_id,
                    "author": r.author,
                    "rating": r.rating,
                    "title": r.title,
                    "comment": r.comment
                }
                for r in reviews
            ]
            for wf_id, reviews in self.reviews.items()
        }

        with open(reviews_file, 'w') as f:
            json.dump(reviews_data, f, indent=2)

        # Save installations
        installations_file = self.storage_path / "installations.json"
        installations_data = {
            wf_id: {
                "workflow_id": inst.workflow_id,
                "version": inst.version,
                "installed_by": inst.installed_by,
                "auto_update": inst.auto_update
            }
            for wf_id, inst in self.installations.items()
        }

        with open(installations_file, 'w') as f:
            json.dump(installations_data, f, indent=2)

    def _load_marketplace(self):
        """Load marketplace data"""
        try:
            templates_file = self.storage_path / "templates.json"
            if templates_file.exists():
                with open(templates_file, 'r') as f:
                    templates_data = json.load(f)
                    self.stats["total_workflows"] = len(templates_data)

            logging.info(f"Loaded {self.stats['total_workflows']} workflows from marketplace")

        except Exception as e:
            logging.error(f"Failed to load marketplace: {e}")


# Demo
async def demo_marketplace():
    """Demonstrate workflow marketplace"""

    print("🏪 Workflow Marketplace Demo\n")

    marketplace = WorkflowMarketplace()

    # Publish a workflow
    print("1️⃣  Publishing workflow...")

    dsl = """
LOOP incident_response
  INPUT: incident_id
  STEPS:
    - assess = assess_severity(incident_id)
    - investigate = root_cause_analysis(incident_id)
    - remediate = auto_remediate(investigate)
  OUTPUT: {assess, investigate, remediate}
END
"""

    workflow_id = await marketplace.publish(
        dsl=dsl,
        metadata={
            "workflow_id": "incident-response-auto",
            "name": "Automated Incident Response",
            "version": "1.0.0",
            "author": "devops-team",
            "description": "Automated workflow for incident response",
            "category": "incident-response",
            "tags": ["incident", "automation", "ops"]
        },
        documentation="# Incident Response Workflow\n\nAutomated incident handling..."
    )

    print(f"   Published: {workflow_id}")
    print()

    # Search
    print("2️⃣  Searching workflows...")
    results = marketplace.search(query="incident", min_rating=0)
    print(f"   Found {len(results)} workflows")
    print()

    # Install
    print("3️⃣  Installing workflow...")
    result = await marketplace.install(workflow_id, user_id="alice")
    print(f"   Installed: {result['workflow_id']} v{result['version']}")
    print()

    # Rate
    print("4️⃣  Rating workflow...")
    review_id = marketplace.rate(
        workflow_id,
        rating=5,
        title="Excellent workflow!",
        comment="Saves so much time during incidents",
        author="alice"
    )
    print(f"   Review ID: {review_id}")
    print()

    # Statistics
    print("5️⃣  Marketplace Statistics:")
    stats = marketplace.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     - {k}: {v}")
        elif isinstance(value, list):
            print(f"   {key}: {len(value)} items")
        else:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_marketplace())
