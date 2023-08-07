from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

with open("README.md") as fh:
    long_description = fh.read()


def find_requirements(filename):
    with (ROOT / "requirements" / filename).open() as f:
        return [s for s in [line.strip(" \n") for line in f] if not s.startswith("#") and s != ""]


runtime_requires = find_requirements("requirements.txt")
dev_requires = find_requirements("requirements-dev.txt")
docs_require = find_requirements("requirements-docs.txt")


setup(
    name="pix2pix",
    version="0.0.0",
    author="Francisco Caetano",
    description="A short description of the project. No quotes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9.0",
    install_requires=runtime_requires,
    extras_require={
        "dev": dev_requires,
        "docs": docs_require,
    },
)
