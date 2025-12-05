"""
Skill Marketplace CLI
=====================

**Created**: November 22, 2025
**Purpose**: Command-line interface for skill discovery and management

Commands:
- search <query> - Search for skills
- list [--category] [--sort-by] - List all skills
- show <skill-id> - Show skill details
- install <skill-id> - Install a skill
- upgrade <skill-id> - Upgrade to newer version
- remove <skill-id> - Remove installed skill
- installed - List installed skills
- updates - Check for available updates
- stats - Show marketplace statistics

Usage:
```bash
python -m skills.marketplace.cli search "code review"
python -m skills.marketplace.cli install code_reviewer:1.0.0
python -m skills.marketplace.cli list --category=meta
python -m skills.marketplace.cli stats
```
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from skills.marketplace.catalog import (
    SkillCatalog,
    SkillMetadata,
    SkillCategory,
    SortBy,
    SearchResult
)
from skills.marketplace.package_manager import PackageManager

logger = logging.getLogger(__name__)


# ============================================================================
# CLI Commands
# ============================================================================

class MarketplaceCLI:
    """Command-line interface for skill marketplace."""

    def __init__(self, catalog: SkillCatalog, package_manager: PackageManager):
        self.catalog = catalog
        self.package_manager = package_manager

    async def search(self, query: str, category: Optional[str] = None, limit: int = 20):
        """Search for skills."""
        print(f"\nSearching for: '{query}'")

        # Parse category
        cat = SkillCategory(category) if category else None

        # Search
        results = await self.catalog.search(
            query=query,
            category=cat,
            limit=limit
        )

        if not results:
            print("No results found.")
            return

        print(f"\nFound {len(results)} results:\n")

        # Display results
        for i, result in enumerate(results, 1):
            skill = result.skill
            print(f"{i}. {skill.name} (v{skill.version})")
            print(f"   Category: {skill.category.value}")
            print(f"   Description: {skill.description[:80]}...")
            print(f"   Relevance: {result.score:.2f} | Rating: {skill.rating:.1f}/5 ({skill.rating_count} reviews)")
            print(f"   Installs: {skill.install_count} | Usage: {skill.usage_count}")
            print(f"   ID: {skill.skill_id}")
            print()

    async def list_skills(
        self,
        category: Optional[str] = None,
        sort_by: str = "name",
        limit: Optional[int] = None
    ):
        """List all skills."""
        # Parse category
        cat = SkillCategory(category) if category else None

        # Parse sort_by
        try:
            sort = SortBy(sort_by)
        except ValueError:
            print(f"Invalid sort_by: {sort_by}. Using 'name'.")
            sort = SortBy.NAME

        # List skills
        skills = await self.catalog.list_all(
            category=cat,
            sort_by=sort,
            limit=limit
        )

        if not skills:
            print("No skills found.")
            return

        print(f"\n{len(skills)} skills found:\n")

        # Display skills
        for i, skill in enumerate(skills, 1):
            print(f"{i}. {skill.name} (v{skill.version}) - {skill.category.value}")
            print(f"   {skill.description[:80]}...")
            print(f"   Rating: {skill.rating:.1f}/5 | Installs: {skill.install_count}")
            print(f"   ID: {skill.skill_id}")
            print()

    async def show(self, skill_id: str):
        """Show skill details."""
        skill = await self.catalog.get_skill(skill_id)
        if not skill:
            print(f"Skill not found: {skill_id}")
            return

        print(f"\n{'='*70}")
        print(f"Skill: {skill.name} (v{skill.version})")
        print(f"{'='*70}\n")

        print(f"Category: {skill.category.value}")
        print(f"Description: {skill.description}")
        print(f"\nTags: {', '.join(skill.tags)}")
        print(f"Capabilities: {', '.join(skill.capabilities)}")

        print(f"\nRequirements:")
        print(f"  - WarpSpace: {skill.requires_warp_space}")
        print(f"  - YarnGraph: {skill.requires_yarn_graph}")
        print(f"  - EventBus: {skill.requires_event_bus}")
        print(f"  - MissionControl: {skill.requires_mission_control}")
        if skill.requires_tools:
            print(f"  - Tools: {', '.join(skill.requires_tools)}")
        if skill.dependencies:
            print(f"  - Dependencies: {', '.join(skill.dependencies)}")

        print(f"\nAuthor: {skill.author}")
        if skill.homepage:
            print(f"Homepage: {skill.homepage}")
        if skill.repository:
            print(f"Repository: {skill.repository}")
        print(f"License: {skill.license}")

        print(f"\nMetrics:")
        print(f"  - Rating: {skill.rating:.1f}/5 ({skill.rating_count} reviews)")
        print(f"  - Installs: {skill.install_count}")
        print(f"  - Usage: {skill.usage_count}")

        print(f"\nTimestamps:")
        print(f"  - Created: {skill.created_at}")
        print(f"  - Updated: {skill.updated_at}")
        if skill.last_used_at:
            print(f"  - Last used: {skill.last_used_at}")

        print()

    async def install(self, skill_id: str, force: bool = False):
        """Install a skill."""
        print(f"\nInstalling: {skill_id}")

        result = await self.package_manager.install(
            skill_id=skill_id,
            force=force
        )

        if result.success:
            print(f"[OK] {result.message}")
            print(f"     Installed to: {result.installed_path}")
            if result.dependencies_installed:
                print(f"     Dependencies: {', '.join(result.dependencies_installed)}")
            if result.warnings:
                print(f"\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print(f"[FAIL] {result.message}")
            if result.errors:
                print(f"\nErrors:")
                for error in result.errors:
                    print(f"  - {error}")

    async def upgrade(self, skill_id: str, version: Optional[str] = None):
        """Upgrade a skill."""
        print(f"\nUpgrading: {skill_id}")
        if version:
            print(f"Target version: {version}")

        result = await self.package_manager.upgrade(
            skill_id=skill_id,
            version=version
        )

        if result.success:
            print(f"[OK] {result.message}")
            print(f"     {result.old_version} -> {result.new_version}")
            if result.warnings:
                print(f"\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print(f"[FAIL] {result.message}")
            if result.errors:
                print(f"\nErrors:")
                for error in result.errors:
                    print(f"  - {error}")

    async def remove(self, skill_id: str, force: bool = False):
        """Remove a skill."""
        print(f"\nRemoving: {skill_id}")

        # Confirm
        if not force:
            confirm = input(f"Are you sure you want to remove {skill_id}? [y/N]: ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return

        result = await self.package_manager.remove(
            skill_id=skill_id,
            force=force
        )

        if result.success:
            print(f"[OK] {result.message}")
            if result.backup_path:
                print(f"     Backup saved to: {result.backup_path}")
            if result.warnings:
                print(f"\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print(f"[FAIL] {result.message}")
            if result.errors:
                print(f"\nErrors:")
                for error in result.errors:
                    print(f"  - {error}")

    async def list_installed(self):
        """List installed skills."""
        skills = await self.package_manager.list_installed()

        if not skills:
            print("\nNo skills installed.")
            return

        print(f"\n{len(skills)} installed skills:\n")

        for i, skill in enumerate(skills, 1):
            print(f"{i}. {skill.name} (v{skill.version})")
            print(f"   Category: {skill.category.value}")
            print(f"   Usage: {skill.usage_count} times")
            print(f"   ID: {skill.skill_id}")
            print()

    async def check_updates(self):
        """Check for available updates."""
        print("\nChecking for updates...")

        updates = await self.package_manager.check_updates()

        if not updates:
            print("All skills are up to date.")
            return

        print(f"\n{len(updates)} updates available:\n")

        for skill_id, new_version in updates.items():
            skill = await self.catalog.get_skill(skill_id)
            if skill:
                print(f"- {skill.name}: {skill.version} -> {new_version}")

        print(f"\nRun 'upgrade <skill-id>' to update.")

    async def stats(self):
        """Show marketplace statistics."""
        stats = await self.catalog.get_stats()

        print("\n" + "="*70)
        print("Marketplace Statistics")
        print("="*70 + "\n")

        print(f"Total Skills: {stats['total_skills']}")
        print(f"\nBy Category:")
        print(f"  - Meta:    {stats['by_category']['meta']}")
        print(f"  - Domain:  {stats['by_category']['domain']}")
        print(f"  - Agentic: {stats['by_category']['agentic']}")

        print(f"\nTotal Installs: {stats['total_installs']}")
        print(f"Total Usage: {stats['total_usage']}")
        print(f"Average Rating: {stats['avg_rating']:.2f}/5")

        print()


# ============================================================================
# Main CLI
# ============================================================================

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Skill Marketplace CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for skills')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--category', help='Filter by category (meta/domain/agentic)')
    search_parser.add_argument('--limit', type=int, default=20, help='Max results')

    # List command
    list_parser = subparsers.add_parser('list', help='List all skills')
    list_parser.add_argument('--category', help='Filter by category')
    list_parser.add_argument('--sort-by', default='name', help='Sort by (name/popularity/rating/recency)')
    list_parser.add_argument('--limit', type=int, help='Max results')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show skill details')
    show_parser.add_argument('skill_id', help='Skill ID')

    # Install command
    install_parser = subparsers.add_parser('install', help='Install a skill')
    install_parser.add_argument('skill_id', help='Skill ID')
    install_parser.add_argument('--force', action='store_true', help='Force reinstall')

    # Upgrade command
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade a skill')
    upgrade_parser.add_argument('skill_id', help='Skill ID')
    upgrade_parser.add_argument('--version', help='Target version')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a skill')
    remove_parser.add_argument('skill_id', help='Skill ID')
    remove_parser.add_argument('--force', action='store_true', help='Skip confirmation')

    # Installed command
    subparsers.add_parser('installed', help='List installed skills')

    # Updates command
    subparsers.add_parser('updates', help='Check for updates')

    # Stats command
    subparsers.add_parser('stats', help='Show statistics')

    args = parser.parse_args()

    # Initialize catalog and package manager
    catalog = SkillCatalog()
    await catalog.initialize()

    package_manager = PackageManager(catalog)

    cli = MarketplaceCLI(catalog, package_manager)

    try:
        # Execute command
        if args.command == 'search':
            await cli.search(args.query, args.category, args.limit)

        elif args.command == 'list':
            await cli.list_skills(args.category, args.sort_by, args.limit)

        elif args.command == 'show':
            await cli.show(args.skill_id)

        elif args.command == 'install':
            await cli.install(args.skill_id, args.force)

        elif args.command == 'upgrade':
            await cli.upgrade(args.skill_id, args.version)

        elif args.command == 'remove':
            await cli.remove(args.skill_id, args.force)

        elif args.command == 'installed':
            await cli.list_installed()

        elif args.command == 'updates':
            await cli.check_updates()

        elif args.command == 'stats':
            await cli.stats()

        else:
            parser.print_help()

    finally:
        await catalog.close()


if __name__ == '__main__':
    asyncio.run(main())
