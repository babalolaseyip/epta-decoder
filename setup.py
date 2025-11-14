from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="epta-decoder",
    version="0.1.0",
    author="Oluwaseyi Paul Babalola",
    author_email="",
    description="Extended Parity-Check Transformation Algorithm for cyclic codes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/babalolaseyip/epta-decoder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "pytest-cov"],
    },
)
