# Installation script for python
from setuptools import setup, find_packages
import re
import pathlib

HERE = pathlib.Path(__file__).parent.absolute()
PACKAGE = "qibochem"


# Returns the qibotn version
def version():
    """Gets the version from the package's __init__ file
    if there is some problem, let it happily fail"""
    version_file = HERE / "src" / PACKAGE / "__init__.py"
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"

    initfile = version_file.read_text(encoding="utf-8")
    matched = re.search(version_regex, initfile, re.M)

    if matched is not None:
        return matched.group(1)
    return "0.0.0"


# load long description from README
setup(
    name="qibochem",
    version=version(),
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
        "qibo>=0.1.10",
        "openfermion",
        "pyscf",
    ],
    # extras_require={
    #     "docs": ["sphinx", "sphinx_rtd_theme", "recommonmark", "sphinxcontrib-bibtex", "sphinx_markdown_tables", "nbsphinx", "IPython", "doc2dash>=2.4.1", ],
    #     "tests": ["pytest", "cirq", "ply", "sklearn", "dill", "coverage", "pytest-cov"],
    # }
    extras_require={
        "docs": [],
        "tests": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "pytest-env>=0.8.1",
        ],
        "analysis": [
            "pylint>=2.16.0",
        ],
    },
    python_requires=">=3.7.0",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)
