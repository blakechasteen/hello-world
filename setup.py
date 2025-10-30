"""
HoloLoom Setup
==============

An AI assistant that learns from you.

Installation:
    pip install -e .  # Development mode
    pip install .     # Production install
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        # Skip comments, empty lines, and optional dependencies
        if line and not line.startswith("#") and not line.startswith("//"):
            # Extract package name (before comments)
            if "#" in line:
                line = line.split("#")[0].strip()
            if line:
                requirements.append(line)

setup(
    name="hololoom",
    version="1.0.0",
    author="Blake Chasteen",
    author_email="blakechasteen@users.noreply.github.com",
    description="An AI assistant that learns from you - self-improving with recursive learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blakechasteen/mythRL",
    project_urls={
        "Bug Tracker": "https://github.com/blakechasteen/mythRL/issues",
        "Documentation": "https://github.com/blakechasteen/mythRL/blob/master/README.md",
        "Source Code": "https://github.com/blakechasteen/mythRL",
        "Release Notes": "https://github.com/blakechasteen/mythRL/blob/master/RELEASE_v1.0.0.md",
    },
    packages=find_packages(include=["HoloLoom", "HoloLoom.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "nlp": [
            "spacy>=3.7.0,<4.0.0",
        ],
        "production": [
            "qdrant-client>=1.7.0,<2.0.0",
            "neo4j>=5.0.0,<6.0.0",
        ],
        "dev": [
            "pytest>=7.4.0,<8.0.0",
            "pytest-asyncio>=0.21.0,<1.0.0",
            "pytest-cov>=4.1.0,<5.0.0",
            "black>=23.0.0,<24.0.0",
            "mypy>=1.5.0,<2.0.0",
            "ruff>=0.1.0,<1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0,<4.0.0",
            "plotly>=5.17.0,<6.0.0",
        ],
        "all": [
            "spacy>=3.7.0,<4.0.0",
            "qdrant-client>=1.7.0,<2.0.0",
            "neo4j>=5.0.0,<6.0.0",
            "scipy>=1.10.0,<2.0.0",
            "matplotlib>=3.7.0,<4.0.0",
            "plotly>=5.17.0,<6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hololoom=HoloLoom.cli:main",  # Future: CLI interface
        ],
    },
    include_package_data=True,
    package_data={
        "HoloLoom": [
            "*.md",
            "config/*.yaml",
            "visualization/*.css",
            "visualization/*.js",
        ],
    },
    keywords=[
        "ai",
        "machine-learning",
        "recursive-learning",
        "thompson-sampling",
        "knowledge-graph",
        "memory-system",
        "neurosymbolic",
        "self-improvement",
        "provenance",
        "graphrag",
    ],
    zip_safe=False,
)