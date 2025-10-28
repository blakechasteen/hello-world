from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mythy",
    version="0.1.0",
    description="Narrative intelligence system built on HoloLoom - Joseph Campbell, depth analysis, cross-domain adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="mythRL Team",
    author_email="contact@mythr.ai",
    url="https://github.com/blakechasteen/mythRL",

    packages=find_packages(),

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

    keywords="narrative analysis joseph-campbell heros-journey mythology storytelling nlp ai framework mythr",

    project_urls={
        "Bug Reports": "https://github.com/blakechasteen/mythRL/issues",
        "Source": "https://github.com/blakechasteen/mythRL",
    },
)
