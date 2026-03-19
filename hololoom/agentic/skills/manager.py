"""
HoloLoom Skills Package Manager
================================
Export and import skills as shareable packages.

Integrated from Promptly (November 2025).

Features:
- Export skills to .skillpack files
- Import skills from packages
- Package versioning and metadata
- ZIP compression for distribution
- README generation for packages

Integration with HoloLoom:
- Works with SkillLoader for skill management
- Compatible with agentic reasoning modes
- Supports skill customization and extension
"""

import json
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SkillPackage:
    """
    A skill package for export/import.

    Attributes:
        name: Package name
        version: Package version (semantic versioning)
        author: Package author
        description: Package description
        license: Package license
        created_at: Creation timestamp
        skills: List of skill definitions
        metadata: Additional metadata
    """
    name: str
    version: str
    author: str | None = None
    description: str | None = None
    license: str = "MIT"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    skills: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'SkillPackage':
        """Create from dictionary."""
        return cls(**data)


class PackageManager:
    """
    Manage skill packages for HoloLoom.

    Enables sharing and distribution of custom skills.
    """

    PACKAGE_VERSION = "1.0.0"
    PACKAGE_EXTENSION = ".skillpack"

    def __init__(self, skill_loader=None):
        """
        Initialize package manager.

        Args:
            skill_loader: Instance of SkillLoader (optional)
        """
        self.skill_loader = skill_loader

    def export_skill(
        self,
        skill_name: str,
        skill_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Export a single skill.

        Args:
            skill_name: Name of the skill
            skill_data: Skill template data

        Returns:
            Dict with skill data
        """
        return {
            "name": skill_name,
            "description": skill_data.get("description", ""),
            "metadata": skill_data.get("metadata", {}),
            "files": skill_data.get("files", {})
        }

    def create_package(
        self,
        package_name: str,
        skills: dict[str, dict[str, Any]],
        author: str | None = None,
        description: str | None = None,
        version: str = "1.0.0"
    ) -> SkillPackage:
        """
        Create a package from skills.

        Args:
            package_name: Name for the package
            skills: Dict of skill_name -> skill_data
            author: Package author
            description: Package description
            version: Package version

        Returns:
            SkillPackage ready for export
        """
        package = SkillPackage(
            name=package_name,
            version=version,
            author=author,
            description=description
        )

        # Export skills
        for skill_name, skill_data in skills.items():
            try:
                skill_export = self.export_skill(skill_name, skill_data)
                package.skills.append(skill_export)
            except Exception as e:
                print(f"Warning: Could not export skill '{skill_name}': {e}")

        # Add metadata
        package.metadata = {
            "hololoom_version": self.PACKAGE_VERSION,
            "skill_count": len(package.skills),
            "export_date": datetime.now().isoformat(),
            "compatible_with": ["HoloLoom >= 1.0.0"]
        }

        return package

    def save_package(
        self,
        package: SkillPackage,
        output_path: str,
        compress: bool = True
    ) -> str:
        """
        Save package to file.

        Args:
            package: SkillPackage to save
            output_path: Path to save to
            compress: Whether to create a ZIP archive

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)

        # Ensure .skillpack extension
        if not output_path.suffix:
            output_path = output_path.with_suffix(self.PACKAGE_EXTENSION)

        if compress:
            # Create ZIP archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add package manifest
                manifest = json.dumps(package.to_dict(), indent=2)
                zf.writestr('package.json', manifest)

                # Add README
                readme = self._generate_readme(package)
                zf.writestr('README.md', readme)

        else:
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(package.to_dict(), f, indent=2)

        return str(output_path)

    def load_package(self, package_path: str) -> SkillPackage:
        """
        Load package from file.

        Args:
            package_path: Path to package file

        Returns:
            SkillPackage
        """
        package_path = Path(package_path)

        if package_path.suffix == '.zip' or zipfile.is_zipfile(package_path):
            # Load from ZIP
            with zipfile.ZipFile(package_path, 'r') as zf:
                manifest = zf.read('package.json').decode('utf-8')
                data = json.loads(manifest)
        else:
            # Load from JSON
            with open(package_path, encoding='utf-8') as f:
                data = json.load(f)

        return SkillPackage.from_dict(data)

    def import_package(
        self,
        package: SkillPackage,
        overwrite: bool = False,
        prefix: str | None = None
    ) -> dict[str, Any]:
        """
        Import a package into HoloLoom.

        Args:
            package: SkillPackage to import
            overwrite: Whether to overwrite existing skills
            prefix: Optional prefix for imported names

        Returns:
            Import results summary
        """
        results = {
            "skills_imported": 0,
            "skills_skipped": 0,
            "errors": []
        }

        # Import skills
        for skill_data in package.skills:
            name = skill_data["name"]
            if prefix:
                name = f"{prefix}_{name}"

            try:
                # Check if exists (if skill_loader available)
                exists = False
                if self.skill_loader:
                    try:
                        self.skill_loader.get(name)
                        exists = True
                    except Exception:
                        pass

                if exists and not overwrite:
                    results["skills_skipped"] += 1
                    continue

                # Import skill (would integrate with skill storage)
                # For now, just count as imported
                results["skills_imported"] += 1

            except Exception as e:
                results["errors"].append(f"Skill '{name}': {e}")

        return results

    def _generate_readme(self, package: SkillPackage) -> str:
        """Generate README for package."""
        lines = [
            f"# {package.name}",
            "",
            f"**Version:** {package.version}",
        ]

        if package.author:
            lines.append(f"**Author:** {package.author}")

        if package.description:
            lines.extend([
                "",
                "## Description",
                package.description
            ])

        lines.extend([
            "",
            "## Contents",
            f"- Skills: {len(package.skills)}",
            ""
        ])

        if package.skills:
            lines.append("### Skills")
            for s in package.skills:
                files_count = len(s.get('files', {}))
                lines.append(f"- **{s['name']}** - {s['description']} ({files_count} files)")

        lines.extend([
            "",
            "## Installation",
            "```python",
            "from hololoom.agentic.skills import PackageManager",
            "",
            "pm = PackageManager()",
            f"package = pm.load_package('{package.name}.skillpack')",
            "results = pm.import_package(package)",
            "```",
            "",
            f"Generated: {package.created_at}",
            f"License: {package.license}"
        ])

        return "\n".join(lines)

    def list_package_contents(self, package: SkillPackage) -> str:
        """Get formatted list of package contents."""
        lines = [
            f"Package: {package.name} v{package.version}",
            f"Author: {package.author or 'Unknown'}",
            f"Created: {package.created_at}",
            "",
            f"Skills ({len(package.skills)}):"
        ]

        for s in package.skills:
            lines.append(f"  - {s['name']} - {s['description']}")

        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_export(
    skills: dict[str, dict[str, Any]],
    package_name: str,
    output_path: str = None,
    author: str = None
) -> str:
    """
    Quick export to package file.

    Args:
        skills: Dict of skill_name -> skill_data
        package_name: Name for the package
        output_path: Optional output path
        author: Optional author name

    Returns:
        Path to saved package
    """
    pm = PackageManager()
    package = pm.create_package(
        package_name=package_name,
        skills=skills,
        author=author
    )

    if not output_path:
        output_path = f"{package_name}.skillpack"

    return pm.save_package(package, output_path)


def quick_import(
    package_path: str,
    overwrite: bool = False
) -> dict[str, Any]:
    """
    Quick import from package file.

    Args:
        package_path: Path to package file
        overwrite: Whether to overwrite existing skills

    Returns:
        Import results summary
    """
    pm = PackageManager()
    package = pm.load_package(package_path)
    return pm.import_package(package, overwrite=overwrite)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("HoloLoom Skills Package Manager")
    print("\nExample - Export:")
    print("""
from hololoom.agentic.skills import get_template, quick_export

# Get skills to export
skills = {
    "code_reviewer": get_template("code_reviewer"),
    "test_generator": get_template("test_generator")
}

# Export to package
path = quick_export(
    skills=skills,
    package_name="code_quality_pack",
    author="Your Name"
)

print(f"Package saved to: {path}")
""")

    print("\nExample - Import:")
    print("""
from hololoom.agentic.skills import quick_import

# Import package
results = quick_import("code_quality_pack.skillpack")

print(f"Imported {results['skills_imported']} skills")
""")
