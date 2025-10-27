#!/usr/bin/env python3
"""
Promptly Package Manager
=========================
Export and import prompts, skills, and collections as shareable packages.
"""

import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
import base64


@dataclass
class PromptPackage:
    """A prompt package for export/import"""
    name: str
    version: str
    author: Optional[str] = None
    description: Optional[str] = None
    license: str = "MIT"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PromptPackage':
        """Create from dictionary"""
        return cls(**data)


class PackageManager:
    """Manage prompt packages"""

    PACKAGE_VERSION = "1.0.0"
    PACKAGE_EXTENSION = ".promptly"

    def __init__(self, promptly_instance):
        """
        Initialize package manager.

        Args:
            promptly_instance: Instance of Promptly
        """
        self.promptly = promptly_instance

    def export_prompt(
        self,
        prompt_name: str,
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export a single prompt.

        Returns:
            Dict with prompt data
        """
        prompt_data = self.promptly.get(prompt_name, version=version)

        return {
            "name": prompt_data["name"],
            "content": prompt_data["content"],
            "version": prompt_data["version"],
            "metadata": prompt_data.get("metadata", {})
        }

    def export_skill(
        self,
        skill_name: str,
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export a skill with all files.

        Returns:
            Dict with skill data and files
        """
        skill_data = self.promptly.get_skill(skill_name, version=version)
        files = self.promptly.get_skill_files(skill_name)

        return {
            "name": skill_data["name"],
            "description": skill_data["description"],
            "version": skill_data["version"],
            "metadata": skill_data.get("metadata", {}),
            "files": [
                {
                    "filename": f["filename"],
                    "filetype": f["filetype"],
                    "content": f["content"]
                }
                for f in files
            ]
        }

    def create_package(
        self,
        package_name: str,
        prompt_names: Optional[List[str]] = None,
        skill_names: Optional[List[str]] = None,
        author: Optional[str] = None,
        description: Optional[str] = None,
        version: str = "1.0.0"
    ) -> PromptPackage:
        """
        Create a package from prompts and skills.

        Args:
            package_name: Name for the package
            prompt_names: List of prompt names to include
            skill_names: List of skill names to include
            author: Package author
            description: Package description
            version: Package version

        Returns:
            PromptPackage ready for export
        """
        package = PromptPackage(
            name=package_name,
            version=version,
            author=author,
            description=description
        )

        # Export prompts
        if prompt_names:
            for name in prompt_names:
                try:
                    prompt = self.export_prompt(name)
                    package.prompts.append(prompt)
                except Exception as e:
                    print(f"Warning: Could not export prompt '{name}': {e}")

        # Export skills
        if skill_names:
            for name in skill_names:
                try:
                    skill = self.export_skill(name)
                    package.skills.append(skill)
                except Exception as e:
                    print(f"Warning: Could not export skill '{name}': {e}")

        # Add metadata
        package.metadata = {
            "promptly_version": self.PACKAGE_VERSION,
            "prompt_count": len(package.prompts),
            "skill_count": len(package.skills),
            "export_date": datetime.now().isoformat()
        }

        return package

    def save_package(
        self,
        package: PromptPackage,
        output_path: str,
        compress: bool = True
    ) -> str:
        """
        Save package to file.

        Args:
            package: PromptPackage to save
            output_path: Path to save to
            compress: Whether to create a ZIP archive

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)

        # Ensure .promptly extension
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

    def load_package(self, package_path: str) -> PromptPackage:
        """
        Load package from file.

        Args:
            package_path: Path to package file

        Returns:
            PromptPackage
        """
        package_path = Path(package_path)

        if package_path.suffix == '.zip' or zipfile.is_zipfile(package_path):
            # Load from ZIP
            with zipfile.ZipFile(package_path, 'r') as zf:
                manifest = zf.read('package.json').decode('utf-8')
                data = json.loads(manifest)
        else:
            # Load from JSON
            with open(package_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        return PromptPackage.from_dict(data)

    def import_package(
        self,
        package: PromptPackage,
        overwrite: bool = False,
        prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import a package into Promptly.

        Args:
            package: PromptPackage to import
            overwrite: Whether to overwrite existing prompts/skills
            prefix: Optional prefix for imported names

        Returns:
            Import results summary
        """
        results = {
            "prompts_imported": 0,
            "prompts_skipped": 0,
            "skills_imported": 0,
            "skills_skipped": 0,
            "errors": []
        }

        # Import prompts
        for prompt_data in package.prompts:
            name = prompt_data["name"]
            if prefix:
                name = f"{prefix}_{name}"

            try:
                # Check if exists
                exists = False
                try:
                    self.promptly.get(name)
                    exists = True
                except:
                    pass

                if exists and not overwrite:
                    results["prompts_skipped"] += 1
                    continue

                # Import
                self.promptly.add(
                    name=name,
                    content=prompt_data["content"],
                    metadata=prompt_data.get("metadata", {})
                )
                results["prompts_imported"] += 1

            except Exception as e:
                results["errors"].append(f"Prompt '{name}': {e}")

        # Import skills
        for skill_data in package.skills:
            name = skill_data["name"]
            if prefix:
                name = f"{prefix}_{name}"

            try:
                # Check if exists
                exists = False
                try:
                    self.promptly.get_skill(name)
                    exists = True
                except:
                    pass

                if exists and not overwrite:
                    results["skills_skipped"] += 1
                    continue

                # Import skill
                self.promptly.add_skill(
                    name=name,
                    description=skill_data["description"],
                    metadata=skill_data.get("metadata", {})
                )

                # Import files
                import tempfile
                for file_data in skill_data.get("files", []):
                    # Create temp file
                    with tempfile.NamedTemporaryFile(
                        mode='w',
                        delete=False,
                        suffix=f'.{file_data["filetype"]}',
                        encoding='utf-8'
                    ) as tmp:
                        tmp.write(file_data["content"])
                        tmp_path = tmp.name

                    try:
                        self.promptly.add_skill_file(
                            name,
                            tmp_path,
                            filetype=file_data["filetype"]
                        )
                    finally:
                        Path(tmp_path).unlink()

                results["skills_imported"] += 1

            except Exception as e:
                results["errors"].append(f"Skill '{name}': {e}")

        return results

    def _generate_readme(self, package: PromptPackage) -> str:
        """Generate README for package"""
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
            f"- Prompts: {len(package.prompts)}",
            f"- Skills: {len(package.skills)}",
            ""
        ])

        if package.prompts:
            lines.append("### Prompts")
            for p in package.prompts:
                lines.append(f"- **{p['name']}** (v{p['version']})")

        if package.skills:
            lines.append("")
            lines.append("### Skills")
            for s in package.skills:
                files_count = len(s.get('files', []))
                lines.append(f"- **{s['name']}** - {s['description']} ({files_count} files)")

        lines.extend([
            "",
            "## Installation",
            "```python",
            "from promptly import Promptly",
            "from package_manager import PackageManager",
            "",
            "p = Promptly()",
            "pm = PackageManager(p)",
            f"package = pm.load_package('{package.name}.promptly')",
            "results = pm.import_package(package)",
            "```",
            "",
            f"Generated: {package.created_at}",
            f"License: {package.license}"
        ])

        return "\n".join(lines)

    def list_package_contents(self, package: PromptPackage) -> str:
        """Get formatted list of package contents"""
        lines = [
            f"Package: {package.name} v{package.version}",
            f"Author: {package.author or 'Unknown'}",
            f"Created: {package.created_at}",
            "",
            f"Prompts ({len(package.prompts)}):"
        ]

        for p in package.prompts:
            lines.append(f"  - {p['name']} (v{p['version']})")

        lines.append(f"\nSkills ({len(package.skills)}):")
        for s in package.skills:
            lines.append(f"  - {s['name']} - {s['description']}")

        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_export(
    promptly_instance,
    package_name: str,
    prompts: List[str] = None,
    skills: List[str] = None,
    output_path: str = None
) -> str:
    """Quick export to package file"""
    pm = PackageManager(promptly_instance)
    package = pm.create_package(
        package_name=package_name,
        prompt_names=prompts,
        skill_names=skills
    )

    if not output_path:
        output_path = f"{package_name}.promptly"

    return pm.save_package(package, output_path)


def quick_import(
    promptly_instance,
    package_path: str,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Quick import from package file"""
    pm = PackageManager(promptly_instance)
    package = pm.load_package(package_path)
    return pm.import_package(package, overwrite=overwrite)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Promptly Package Manager")
    print("\nExample - Export:")
    print("""
from promptly import Promptly
from package_manager import quick_export

p = Promptly()

# Export prompts and skills to package
path = quick_export(
    p,
    package_name="my_collection",
    prompts=["summarizer", "analyzer"],
    skills=["code_reviewer"]
)

print(f"Package saved to: {path}")
""")

    print("\nExample - Import:")
    print("""
from promptly import Promptly
from package_manager import quick_import

p = Promptly()

# Import package
results = quick_import(p, "my_collection.promptly")

print(f"Imported {results['prompts_imported']} prompts")
print(f"Imported {results['skills_imported']} skills")
""")
