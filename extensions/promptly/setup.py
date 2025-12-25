#!/usr/bin/env python3
"""
Promptly - Personal Prompt Management Suite

Installation:
    pip install -e .

This will install the 'promptly' command globally.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="promptly",
    version="1.0.0",
    author="Blake",
    author_email="blake@example.com",
    description="Personal prompt management suite with git-like versioning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/promptly",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "mcp": ["mcp>=0.1.0"],
        "rich": ["rich>=13.0.0"],
        "anthropic": ["anthropic>=0.34.0"],
        "ollama": ["ollama>=0.1.0"],
        "all": [
            "mcp>=0.1.0",
            "rich>=13.0.0",
            "anthropic>=0.34.0",
            "ollama>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "promptly=promptly.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing",
    ],
    keywords="prompt management cli versioning llm",
)
