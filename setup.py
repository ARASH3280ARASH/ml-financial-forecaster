"""Setup configuration for ml-financial-forecaster."""

from pathlib import Path

from setuptools import find_packages, setup

README = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""

setup(
    name="ml-financial-forecaster",
    version="1.0.0",
    author="ARASH3280ARASH",
    description="Production-grade ML pipeline for financial time series forecasting",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ARASH3280ARASH/ml-financial-forecaster",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "deep-learning": ["tensorflow>=2.14.0"],
        "tuning": ["optuna>=3.4.0"],
        "stats": ["statsmodels>=0.14.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "all": [
            "tensorflow>=2.14.0",
            "optuna>=3.4.0",
            "statsmodels>=0.14.0",
            "seaborn>=0.13.0",
            "pytest>=7.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
