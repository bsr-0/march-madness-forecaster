from setuptools import setup, find_packages

setup(
    name="march-madness-forecaster",
    version="0.1.0",
    description="Mathematically robust prediction system for NCAA March Madness tournament",
    author="Ben Rosen",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.4,<2.0.0",
        "pandas>=1.5.3,<2.0.0",
        "pytz>=2022.7,<2024.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "cbbpy==2.0.2",
        "sportsipy>=0.6.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "march-madness=src.main:main",
        ],
    },
)
