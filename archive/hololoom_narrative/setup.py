from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hololoom-narrative",
    version="0.1.0",
    description="Comprehensive narrative intelligence system built on HoloLoom framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/you/hololoom-narrative",

    packages=find_packages(include=["hololoom_narrative", "hololoom_narrative.*"]),

    install_requires=[
        # Framework dependency (required)
        # Note: Currently expects hololoom to be available locally
        # In production: "hololoom>=0.1.0"
    ],

    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
        ],
    },

    python_requires=">=3.9",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],

    keywords="narrative analysis joseph-campbell heros-journey mythology storytelling nlp ai framework",

    project_urls={
        "Bug Reports": "https://github.com/you/hololoom-narrative/issues",
        "Source": "https://github.com/you/hololoom-narrative",
        "Documentation": "https://github.com/you/hololoom-narrative/tree/main/docs",
    },
)
