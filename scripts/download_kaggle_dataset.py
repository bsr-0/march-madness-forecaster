"""Download the nishaanamin/march-madness-data dataset from Kaggle.

Uses kagglehub to download the latest version and copies the files
into data/kaggle/ for use by the pipeline.

Authentication: requires KAGGLE_USERNAME + KAGGLE_KEY env vars,
or a ~/.kaggle/kaggle.json credentials file.

Usage:
    python scripts/download_kaggle_dataset.py
    python scripts/download_kaggle_dataset.py --force   # re-download even if exists
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_SLUG = "nishaanamin/march-madness-data"
OUTPUT_DIR = Path("data/kaggle")


def download_dataset(force: bool = False) -> Path | None:
    """Download the March Madness dataset via kagglehub.

    Args:
        force: Re-download even if data/kaggle/ already has CSV files.

    Returns:
        Path to the output directory, or None on failure.
    """
    if not force and OUTPUT_DIR.exists() and list(OUTPUT_DIR.glob("*.csv")):
        logger.info("Dataset already exists in %s (use --force to re-download)", OUTPUT_DIR)
        return OUTPUT_DIR

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Install it with: pip install kagglehub")
        return None

    logger.info("Downloading dataset: %s", DATASET_SLUG)
    try:
        cached_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    except Exception as e:
        logger.error("Download failed: %s", e)
        return None

    logger.info("Downloaded to cache: %s", cached_path)

    # Copy files into data/kaggle/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_file in cached_path.rglob("*"):
        if src_file.is_file():
            dest = OUTPUT_DIR / src_file.relative_to(cached_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest)
            copied += 1
            logger.info("  %s", dest)

    logger.info("Copied %d files to %s", copied, OUTPUT_DIR)
    return OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Download nishaanamin/march-madness-data from Kaggle")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    args = parser.parse_args()

    result = download_dataset(force=args.force)
    if result:
        csv_files = sorted(result.glob("**/*.csv"))
        print(f"\nDataset ready at: {result}")
        print(f"Files ({len(csv_files)}):")
        for f in csv_files:
            print(f"  {f.relative_to(result)}")
        sys.exit(0)
    else:
        print("Failed to download dataset. Check credentials and network.")
        sys.exit(1)


if __name__ == "__main__":
    main()
