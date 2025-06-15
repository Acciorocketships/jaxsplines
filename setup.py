"""
Setup script for JAXSplines package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for the long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jaxsplines",
    version="0.1.0",
    author="JAXSplines Team",
    author_email="",
    description="B-spline implementation in JAX with Equinox for machine learning applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/jaxsplines",  # Update with actual URL
    packages=find_packages(),
    classifiers=[],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "examples": [
            "matplotlib>=3.0",
            "jupyter",
            "notebook",
        ],
    },
    include_package_data=True,
    package_data={
        "jaxsplines": ["*.py"],
    },
    keywords="jax, splines, b-splines, machine learning, monotonic, invertible",
) 