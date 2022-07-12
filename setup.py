# Installation script for python
from setuptools import setup, find_packages
import os
import re

PACKAGE = "qibochem"

# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qibochem",
    version="0.1",
    description="A chemistry module for quantum computing",
    author="The Qibo team",
    author_email="",
    url="https://github.com/qiboteam/qibochem",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.out", "*.yml"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    install_requires=[
        "pyscf",
        "openfermion",
        "qibo"
    ],
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme", "recommonmark", "sphinxcontrib-bibtex", "sphinx_markdown_tables", "nbsphinx", "IPython", "doc2dash>=2.4.1", ],
        "tests": ["pytest", "cirq", "ply", "sklearn", "dill", "coverage", "pytest-cov"],
    },
    python_requires=">=3.7.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
