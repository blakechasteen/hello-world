#!/usr/bin/env python3
"""
Claude Skills Builder

Packages Claude skills from skill.markdown into .zip/.skill format for deployment.

Usage:
    python scripts/build_skill.py skills/meta/skill_security_analyzer
    python scripts/build_skill.py skills/meta/skill_security_analyzer --validate-only
    python scripts/build_skill.py --all
    python scripts/build_skill.py --help

Features:
    - Validates skill.markdown schema
    - Packages skill directory → .zip
    - Creates .skill format (renamed .zip with metadata)
    - Generates manifest.json
    - Outputs to skills/dist/
    - Optional pre-deployment validation (security, testing)
"""

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SkillMetadata:
    """Parsed metadata from skill.markdown"""
    name: str
    version: str
    author: str
    created: str
    last_updated: str
    category: str
    tags: List[str]
    description_short: str
    description_long: str
    capabilities: List[str]
    dependencies: List[str]


@dataclass
class ValidationResult:
    """Result of skill validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Optional[SkillMetadata]


@dataclass
class BuildResult:
    """Result of skill build"""
    success: bool
    skill_name: str
    output_files: List[str]
    errors: List[str]
    warnings: List[str]


class SkillBuilder:
    """Builds and packages Claude skills"""

    def __init__(self, skills_root: Path, dist_dir: Path):
        self.skills_root = Path(skills_root)
        self.dist_dir = Path(dist_dir)
        self.dist_dir.mkdir(parents=True, exist_ok=True)

    def parse_skill_markdown(self, skill_path: Path) -> ValidationResult:
        """Parse and validate skill.markdown file"""
        errors = []
        warnings = []
        metadata = None

        skill_file = skill_path / "skill.markdown"

        if not skill_file.exists():
            errors.append(f"skill.markdown not found in {skill_path}")
            return ValidationResult(False, errors, warnings, None)

        try:
            content = skill_file.read_text(encoding='utf-8')
        except Exception as e:
            errors.append(f"Failed to read skill.markdown: {e}")
            return ValidationResult(False, errors, warnings, None)

        # Extract metadata section
        metadata_match = re.search(r'## Metadata\s+(.*?)(?=\n## |\Z)', content, re.DOTALL)
        if not metadata_match:
            errors.append("Missing Metadata section")
            return ValidationResult(False, errors, warnings, None)

        metadata_text = metadata_match.group(1)

        # Parse metadata fields
        def extract_field(pattern: str, text: str, required: bool = True) -> Optional[str]:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
            elif required:
                errors.append(f"Missing required field: {pattern}")
            return None

        name = extract_field(r'\*\*Name\*\*:\s*`([^`]+)`', metadata_text)
        version = extract_field(r'\*\*Version\*\*:\s*`([^`]+)`', metadata_text)
        author = extract_field(r'\*\*Author\*\*:\s*`([^`]+)`', metadata_text)
        created = extract_field(r'\*\*Created\*\*:\s*`([^`]+)`', metadata_text)
        last_updated = extract_field(r'\*\*Last Updated\*\*:\s*`([^`]+)`', metadata_text)
        category = extract_field(r'\*\*Category\*\*:\s*`([^`]+)`', metadata_text)
        tags_str = extract_field(r'\*\*Tags\*\*:\s*`([^`]+)`', metadata_text)

        tags = [t.strip() for t in tags_str.split(',')] if tags_str else []

        # Extract description
        desc_match = re.search(r'## Description\s+\*\*Short Description\*\*[:\s]+([^\n]+)', content)
        desc_short = desc_match.group(1).strip() if desc_match else None

        desc_long_match = re.search(r'\*\*Detailed Description\*\*:\s+(.*?)(?=\n## |\Z)', content, re.DOTALL)
        desc_long = desc_long_match.group(1).strip() if desc_long_match else None

        # Extract capabilities
        capabilities = []
        cap_section = re.search(r'## Required Capabilities\s+(.*?)(?=\n## |\Z)', content, re.DOTALL)
        if cap_section:
            cap_text = cap_section.group(1)
            cap_matches = re.findall(r'- \[x\] ([^\n]+)', cap_text)
            capabilities = [cap.strip() for cap in cap_matches]

        # Extract dependencies
        dependencies = []
        dep_section = re.search(r'## Dependencies\s+(.*?)(?=\n## |\Z)', content, re.DOTALL)
        if dep_section:
            dep_text = dep_section.group(1)
            dep_matches = re.findall(r'\*\*Required Skills\*\*.*?`([^`]+)`', dep_text)
            dependencies = [dep.strip() for dep in dep_matches]

        # Validate required sections
        required_sections = [
            "Input Schema",
            "Output Schema",
            "Prompt Template",
            "Examples",
            "Testing Checklist"
        ]

        for section in required_sections:
            if f"## {section}" not in content:
                warnings.append(f"Missing recommended section: {section}")

        # Validate examples exist
        examples_match = re.search(r'## Examples\s+(.*?)(?=\n## |\Z)', content, re.DOTALL)
        if examples_match:
            examples_text = examples_match.group(1)
            example_count = len(re.findall(r'### Example \d+:', examples_text))
            if example_count < 2:
                warnings.append(f"Only {example_count} example(s) found, recommend at least 2-3")
        else:
            warnings.append("No examples found")

        if errors:
            return ValidationResult(False, errors, warnings, None)

        metadata = SkillMetadata(
            name=name or "unknown",
            version=version or "0.0.0",
            author=author or "Unknown",
            created=created or "Unknown",
            last_updated=last_updated or datetime.now().strftime("%Y-%m-%d"),
            category=category or "unknown",
            tags=tags,
            description_short=desc_short or "",
            description_long=desc_long or "",
            capabilities=capabilities,
            dependencies=dependencies
        )

        valid = len(errors) == 0
        return ValidationResult(valid, errors, warnings, metadata)

    def generate_manifest(self, metadata: SkillMetadata, skill_path: Path) -> Dict:
        """Generate manifest.json for skill package"""
        return {
            "format_version": "1.0",
            "skill": {
                "name": metadata.name,
                "version": metadata.version,
                "author": metadata.author,
                "category": metadata.category,
                "tags": metadata.tags,
                "description": metadata.description_short,
                "capabilities": metadata.capabilities,
                "dependencies": metadata.dependencies
            },
            "package": {
                "created_at": datetime.now().isoformat(),
                "source_path": str(skill_path),
                "builder_version": "1.0.0"
            }
        }

    def build_skill(self, skill_path: Path, validate_only: bool = False) -> BuildResult:
        """Build and package a skill"""
        skill_path = Path(skill_path)
        errors = []
        warnings = []
        output_files = []

        # Validate skill.markdown
        validation = self.parse_skill_markdown(skill_path)
        errors.extend(validation.errors)
        warnings.extend(validation.warnings)

        if not validation.valid:
            return BuildResult(False, skill_path.name, [], errors, warnings)

        skill_name = validation.metadata.name

        if validate_only:
            print(f"✓ Validation passed for {skill_name}")
            if warnings:
                print(f"  Warnings: {len(warnings)}")
                for w in warnings:
                    print(f"    - {w}")
            return BuildResult(True, skill_name, [], errors, warnings)

        # Generate manifest
        manifest = self.generate_manifest(validation.metadata, skill_path)
        manifest_path = skill_path / "manifest.json"

        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            errors.append(f"Failed to write manifest.json: {e}")
            return BuildResult(False, skill_name, [], errors, warnings)

        # Create .zip package
        zip_filename = f"{skill_name}-{validation.metadata.version}.zip"
        zip_path = self.dist_dir / zip_filename

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from skill directory
                for file in skill_path.rglob('*'):
                    if file.is_file():
                        arcname = file.relative_to(skill_path)
                        zipf.write(file, arcname)

            output_files.append(str(zip_path))
        except Exception as e:
            errors.append(f"Failed to create .zip: {e}")
            return BuildResult(False, skill_name, [], errors, warnings)

        # Create .skill package (same as .zip with different extension)
        skill_filename = f"{skill_name}-{validation.metadata.version}.skill"
        skill_package_path = self.dist_dir / skill_filename

        try:
            shutil.copy(zip_path, skill_package_path)
            output_files.append(str(skill_package_path))
        except Exception as e:
            warnings.append(f"Failed to create .skill: {e}")

        success = len(errors) == 0
        return BuildResult(success, skill_name, output_files, errors, warnings)

    def build_all_skills(self, validate_only: bool = False) -> List[BuildResult]:
        """Build all skills in skills/ directory"""
        results = []

        # Find all skill directories (contain skill.markdown)
        for category_dir in self.skills_root.iterdir():
            if category_dir.is_dir() and category_dir.name not in ['templates', 'dist', 'proposals']:
                for skill_dir in category_dir.iterdir():
                    if skill_dir.is_dir() and (skill_dir / "skill.markdown").exists():
                        print(f"\nBuilding {skill_dir.name}...")
                        result = self.build_skill(skill_dir, validate_only)
                        results.append(result)

                        if result.success:
                            print(f"✓ {result.skill_name} built successfully")
                            if result.output_files:
                                for f in result.output_files:
                                    print(f"  → {f}")
                        else:
                            print(f"✗ {result.skill_name} build failed")
                            for e in result.errors:
                                print(f"  Error: {e}")

                        if result.warnings:
                            for w in result.warnings:
                                print(f"  Warning: {w}")

        return results


def print_summary(results: List[BuildResult]):
    """Print build summary"""
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    print(f"Total skills: {total}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")

    if failed > 0:
        print("\nFailed skills:")
        for r in results:
            if not r.success:
                print(f"  - {r.skill_name}")
                for e in r.errors[:3]:  # Show first 3 errors
                    print(f"    → {e}")

    total_warnings = sum(len(r.warnings) for r in results)
    if total_warnings > 0:
        print(f"\nTotal warnings: {total_warnings}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build and package Claude skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build single skill
  python scripts/build_skill.py skills/meta/skill_security_analyzer

  # Validate without building
  python scripts/build_skill.py skills/meta/skill_tester --validate-only

  # Build all skills
  python scripts/build_skill.py --all

  # Build all with validation only
  python scripts/build_skill.py --all --validate-only
        """
    )

    parser.add_argument(
        'skill_path',
        nargs='?',
        help='Path to skill directory (e.g., skills/meta/skill_security_analyzer)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Build all skills in skills/ directory'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate skill.markdown without building packages'
    )

    parser.add_argument(
        '--skills-root',
        default='skills',
        help='Root directory for skills (default: skills/)'
    )

    parser.add_argument(
        '--dist-dir',
        default='skills/dist',
        help='Output directory for built packages (default: skills/dist/)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.skill_path:
        parser.error("Either provide skill_path or use --all")

    # Create builder
    builder = SkillBuilder(args.skills_root, args.dist_dir)

    # Build
    if args.all:
        results = builder.build_all_skills(args.validate_only)
        print_summary(results)
        sys.exit(0 if all(r.success for r in results) else 1)
    else:
        result = builder.build_skill(Path(args.skill_path), args.validate_only)

        if result.success:
            print(f"\n✓ {result.skill_name} built successfully")
            if result.output_files:
                print("Output files:")
                for f in result.output_files:
                    print(f"  → {f}")
            sys.exit(0)
        else:
            print(f"\n✗ {result.skill_name} build failed")
            print("Errors:")
            for e in result.errors:
                print(f"  - {e}")
            if result.warnings:
                print("Warnings:")
                for w in result.warnings:
                    print(f"  - {w}")
            sys.exit(1)


if __name__ == '__main__':
    main()
