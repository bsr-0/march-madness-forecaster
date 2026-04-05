from setuptools import setup, find_packages

setup(
    name="march-madness-forecaster",
    version="0.1.0",
    description="Mathematically robust prediction system for NCAA March Madness tournament",
    author="Ben Rosen",
    packages=find_packages(),
    install_requires=[
        # Core Scientific Computing
        "numpy>=1.24.4,<1.27.0",
        "pandas>=1.5.3,<2.2.0",
        "pytz>=2022.7,<2024.0",
        "scipy>=1.10.1,<1.12.0",
        "scikit-learn>=1.5.0,<1.7.0",
        # Gradient Boosting
        "lightgbm>=4.6.0,<5.0.0",
        "xgboost>=2.0.0,<3.0.0",
        # Web Scraping
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "lxml>=4.9.0",
        "cbbpy==2.0.2",
        "sportsipy>=0.6.0",
        # Kaggle API
        "kaggle>=1.5.16,<1.7.0",
        "kagglehub>=1.0.0,<2.0.0",
        # Validation
        "pydantic>=2.0.0,<3.0.0",
        # Data Processing
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
        # Visualization
        "matplotlib>=3.7.0",
        # YAML parsing
        "PyYAML>=6.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "gpu": [
            "torch>=2.0.0,<3.0.0",
            "torch-geometric>=2.4.0,<3.0.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "march-madness=src.main:main",
        ],
    },
)
