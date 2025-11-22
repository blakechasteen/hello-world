#!/usr/bin/env python3
"""
Quick-start script for Claude Skills creation

Usage:
    python scripts/create_skill.py hololoom_rag_helper --category domain
    python scripts/create_skill.py my_skill --category domain --author "Your Name"
    python scripts/create_skill.py --interactive

Philosophy: Automate the boilerplate, focus on the value.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class SkillCreator:
    """Automated skill creation from template"""

    def __init__(self, skills_root: Path = Path("skills")):
        self.skills_root = skills_root
        self.template_path = skills_root / "templates" / "skill.markdown.template"

    def create_skill(
        self,
        skill_name: str,
        category: str = "domain",
        author: str = "Claude Code User",
        short_description: str = "",
        tags: list[str] = None,
        interactive: bool = False
    ) -> bool:
        """Create a new skill from template"""

        # Validate inputs
        if not self._validate_skill_name(skill_name):
            return False

        if category not in ["meta", "domain", "utility"]:
            print(f"❌ Invalid category: {category}")
            print("   Valid categories: meta, domain, utility")
            return False

        # Create skill directory
        skill_dir = self.skills_root / category / skill_name
        if skill_dir.exists():
            print(f"❌ Skill already exists: {skill_dir}")
            return False

        skill_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {skill_dir}")

        # Read template
        if not self.template_path.exists():
            print(f"❌ Template not found: {self.template_path}")
            return False

        template_content = self.template_path.read_text(encoding="utf-8")

        # Interactive mode
        if interactive:
            short_description = input("Short description (1-2 sentences): ").strip()
            tags_input = input("Tags (comma-separated): ").strip()
            tags = [t.strip() for t in tags_input.split(",")] if tags_input else []

        # Substitute placeholders
        today = datetime.now().strftime("%Y-%m-%d")
        skill_content = template_content.replace("[SKILL_NAME]", self._format_skill_name(skill_name))
        skill_content = skill_content.replace("[skill_identifier]", skill_name)
        skill_content = skill_content.replace("[Your Name]", author)
        skill_content = skill_content.replace("[YYYY-MM-DD]", today)
        skill_content = skill_content.replace("[meta|domain|utility]", category)

        if tags:
            skill_content = skill_content.replace("[tag1, tag2, tag3]", ", ".join(tags))

        if short_description:
            skill_content = skill_content.replace(
                "[Concise description of what this skill does]",
                short_description
            )

        # Write skill.markdown
        skill_file = skill_dir / "skill.markdown"
        skill_file.write_text(skill_content, encoding="utf-8")
        print(f"✅ Created skill file: {skill_file}")

        # Success message
        print(f"\n🎉 Skill '{skill_name}' created successfully!")
        print(f"\n📝 Next steps:")
        print(f"   1. Edit {skill_file}")
        print(f"   2. Fill in all sections (input/output schemas, prompt template, examples)")
        print(f"   3. Validate: python scripts/build_skill.py {skill_dir} --validate-only")
        print(f"   4. Package: python scripts/build_skill.py {skill_dir}")
        print(f"\n📚 See docs/skills_workflow.md for complete workflow")

        return True

    def _validate_skill_name(self, name: str) -> bool:
        """Validate skill name format"""
        if not name:
            print("❌ Skill name cannot be empty")
            return False

        if not name.islower() or not all(c.isalnum() or c == "_" for c in name):
            print("❌ Skill name must be lowercase with underscores only")
            print("   Examples: hololoom_rag_helper, typescript_error_explainer")
            return False

        if name.startswith("_") or name.endswith("_"):
            print("❌ Skill name cannot start or end with underscore")
            return False

        return True

    def _format_skill_name(self, name: str) -> str:
        """Format skill name for display (title case)"""
        return " ".join(word.capitalize() for word in name.split("_"))


def main():
    parser = argparse.ArgumentParser(
        description="Create a new Claude skill from template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/create_skill.py hololoom_rag_helper --category domain

  # With metadata
  python scripts/create_skill.py my_skill \\
    --category domain \\
    --author "Your Name" \\
    --description "Short description" \\
    --tags "python,rag,hololoom"

  # Interactive mode
  python scripts/create_skill.py my_skill --interactive

Categories:
  meta      - Skills that orchestrate other skills
  domain    - Domain-specific task skills
  utility   - General-purpose utility skills
        """
    )

    parser.add_argument(
        "skill_name",
        nargs="?",
        help="Skill name (lowercase with underscores, e.g., hololoom_rag_helper)"
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=["meta", "domain", "utility"],
        default="domain",
        help="Skill category (default: domain)"
    )
    parser.add_argument(
        "--author",
        "-a",
        default="Claude Code User",
        help="Author name (default: Claude Code User)"
    )
    parser.add_argument(
        "--description",
        "-d",
        default="",
        help="Short description (1-2 sentences)"
    )
    parser.add_argument(
        "--tags",
        "-t",
        default="",
        help="Comma-separated tags (e.g., 'python,rag,hololoom')"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode (prompt for metadata)"
    )

    args = parser.parse_args()

    # Interactive mode without skill name
    if args.interactive and not args.skill_name:
        print("🎯 Interactive Skill Creation")
        args.skill_name = input("Skill name (lowercase with underscores): ").strip()
        category_input = input("Category (meta/domain/utility) [domain]: ").strip()
        args.category = category_input if category_input else "domain"
        args.author = input(f"Author [{args.author}]: ").strip() or args.author

    if not args.skill_name:
        parser.print_help()
        sys.exit(1)

    # Parse tags
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    # Create skill
    creator = SkillCreator()
    success = creator.create_skill(
        skill_name=args.skill_name,
        category=args.category,
        author=args.author,
        short_description=args.description,
        tags=tags,
        interactive=args.interactive
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
