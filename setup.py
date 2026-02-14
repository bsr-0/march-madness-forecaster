from setuptools import setup, find_packages

setup(
    name="march-madness-forecaster",
    version="0.1.0",
    description="Mathematically robust prediction system for NCAA March Madness tournament",
    author="Ben Rosen",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "march-madness=src.main:main",
        ],
    },
)
