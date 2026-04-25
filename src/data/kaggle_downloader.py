"""Kaggle competition data downloader for March Machine Learning Mania.

Downloads MMasseyOrdinals and other competition CSV files from the Kaggle
API so that the pipeline can use Massey Ordinals for external rating
composites — the single highest-signal feature in the competition.

Requires either:
  - A ~/.kaggle/kaggle.json credentials file, OR
  - KAGGLE_USERNAME + KAGGLE_KEY environment variables
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Competition slugs for Kaggle March Mania (men's and women's combined since 2024)
COMPETITION_SLUGS = [
    "march-machine-learning-mania-2025",
    "march-machine-learning-mania-2024",
    "march-machine-learning-mania-2026",
    # Older single-gender competitions
    "mens-march-mania-2023",
    "march-machine-learning-mania-2023",
]

# The critical CSV files we need (in priority order)
REQUIRED_FILES = [
    "MMasseyOrdinals",  # Primary target: 50-160+ ranking systems
    "MTeams",  # Team ID resolution
]

OPTIONAL_FILES = [
    "MNCAATourneySeeds",
    "MTeamCoaches",
    "MTeamConferences",
    "MConferences",
    "MSeasons",
    "MRegularSeasonCompactResults",
    "MRegularSeasonDetailedResults",
    "MNCAATourneyCompactResults",
    "MNCAATourneyDetailedResults",
    "MGameCities",
    # Women's equivalents
    "WMasseyOrdinals",
    "WTeams",
    "WNCAATourneySeeds",
]

DEFAULT_KAGGLE_DIR = "data/kaggle"

# Sentinel file written after a failed download attempt to prevent
# retrying on every pipeline run (cleared when force=True).
_DOWNLOAD_FAILED_SENTINEL = ".kaggle_download_attempted"


def kaggle_api_available() -> bool:
    """Check if the Kaggle API is importable and credentials exist."""
    try:
        import kaggle  # noqa: F401

        return True
    except (ImportError, OSError):
        pass

    # Check for credentials even without the package
    username = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USER")
    if username and os.environ.get("KAGGLE_KEY"):
        return True

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _setup_kaggle_credentials() -> bool:
    """Ensure Kaggle API credentials are configured.

    Checks (in order):
    1. KAGGLE_USERNAME + KAGGLE_KEY environment variables
    2. KAGGLE_USER alias (mapped to KAGGLE_USERNAME)
    3. ~/.kaggle/kaggle.json file
    4. .env file in the project root (loads KAGGLE_KEY)

    Returns True if credentials are available.
    """
    # Map KAGGLE_USER alias -> KAGGLE_USERNAME for CI environments
    if not os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_USER"):
        os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_USER"]

    # Already configured via env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    # Check for kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True

    # Try loading from .env file in common project locations
    for env_path in [".env", "../.env", "../../.env"]:
        env_file = Path(env_path)
        if env_file.exists():
            _load_env_file(env_file)
            # Re-check KAGGLE_USER alias after .env load
            if not os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_USER"):
                os.environ["KAGGLE_USERNAME"] = os.environ["KAGGLE_USER"]
            if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
                return True

    return False


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from an .env file into os.environ."""
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if (
                        key
                        and value
                        and value
                        not in (
                            "your_kaggle_api_key_here",
                            "your_kaggle_username_here",
                        )
                    ):
                        os.environ.setdefault(key, value)
    except (OSError, ValueError, UnicodeDecodeError):
        pass


def _kaggle_api_exception():
    """Return the Kaggle ApiException class, or None if unavailable."""
    try:
        from kaggle.rest import ApiException

        return ApiException
    except ImportError:
        return None


def _get_kaggle_api():
    """Get an authenticated Kaggle API instance."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "The 'kaggle' package is required to download competition data. Install it with: pip install kaggle"
        )
    api = KaggleApi()
    api.authenticate()
    return api


def download_competition_data(
    output_dir: str = DEFAULT_KAGGLE_DIR,
    competition: Optional[str] = None,
    force: bool = False,
) -> Optional[str]:
    """Download Kaggle March Mania competition data.

    Downloads and extracts all competition CSV files to ``output_dir``.
    If data already exists (MMasseyOrdinals*.csv found), skips download
    unless ``force=True``.

    Args:
        output_dir: Directory to store downloaded CSVs.
        competition: Specific competition slug. If None, tries known slugs.
        force: Re-download even if data already exists.

    Returns:
        Path to the directory containing CSVs, or None on failure.
    """
    output_path = Path(output_dir)

    # Clear sentinel on force re-download
    if force:
        sentinel = output_path / _DOWNLOAD_FAILED_SENTINEL
        if sentinel.exists():
            sentinel.unlink()

    # Check if data already exists
    if not force and _has_massey_ordinals(output_path):
        logger.info(
            "Kaggle data already exists in %s (use force=True to re-download)",
            output_dir,
        )
        return str(output_path)

    # Ensure credentials
    if not _setup_kaggle_credentials():
        logger.warning(
            "Kaggle API credentials not found. Set KAGGLE_USERNAME and "
            "KAGGLE_KEY environment variables, or create ~/.kaggle/kaggle.json. "
            "See: https://www.kaggle.com/docs/api#authentication"
        )
        return None

    try:
        api = _get_kaggle_api()
    except ImportError as e:
        logger.warning("Kaggle API not available: %s", e)
        return None
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning("Kaggle API authentication failed: %s", e)
        return None

    output_path.mkdir(parents=True, exist_ok=True)

    # Build exception tuple — include Kaggle ApiException when available
    _api_exc = _kaggle_api_exception()
    _catch = (OSError, ValueError, RuntimeError) + ((_api_exc,) if _api_exc else ())

    # Try competition slugs in order
    slugs = [competition] if competition else COMPETITION_SLUGS
    for slug in slugs:
        try:
            logger.info("Attempting to download from competition: %s", slug)
            api.competition_download_files(slug, path=str(output_path), quiet=False)
            # Extract any zip files
            _extract_zips(output_path)
            if _has_massey_ordinals(output_path):
                logger.info(
                    "Successfully downloaded Kaggle data to %s from %s",
                    output_dir,
                    slug,
                )
                return str(output_path)
            logger.info("Downloaded %s but no MMasseyOrdinals found", slug)
        except _catch as e:
            logger.debug("Competition %s download failed: %s", slug, e)
            continue

    # If we got here, try downloading individual files
    for slug in slugs:
        try:
            result = _download_individual_files(api, slug, output_path)
            if result:
                return result
        except _catch as e:
            logger.debug("Individual file download from %s failed: %s", slug, e)
            continue

    logger.warning(
        "Failed to download Kaggle competition data. "
        "Try manually: kaggle competitions download -c march-machine-learning-mania-2025 -p %s",
        output_dir,
    )
    return None


def _download_individual_files(api, competition: str, output_path: Path) -> Optional[str]:
    """Try downloading critical files individually."""
    _api_exc = _kaggle_api_exception()
    _catch = (OSError, ValueError, RuntimeError, zipfile.BadZipFile) + ((_api_exc,) if _api_exc else ())
    downloaded_any = False
    for filename in REQUIRED_FILES + OPTIONAL_FILES:
        try:
            api.competition_download_file(
                competition,
                f"{filename}.csv",
                path=str(output_path),
                quiet=True,
            )
            # Check if it downloaded as a zip
            zip_path = output_path / f"{filename}.csv.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(output_path)
                zip_path.unlink()
            downloaded_any = True
        except _catch:
            continue

    if downloaded_any and _has_massey_ordinals(output_path):
        logger.info("Downloaded individual files to %s", output_path)
        return str(output_path)
    return None


def _extract_zips(directory: Path) -> None:
    """Extract all zip files in a directory."""
    for zip_file in directory.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(directory)
            zip_file.unlink()
            logger.debug("Extracted %s", zip_file.name)
        except Exception as e:
            logger.warning("Failed to extract %s: %s", zip_file, e)


def _has_massey_ordinals(directory: Path) -> bool:
    """Check if the directory contains MMasseyOrdinals CSV data."""
    if not directory.is_dir():
        return False
    for f in directory.iterdir():
        if f.is_file() and "massey" in f.name.lower() and f.suffix == ".csv":
            # Verify it's not empty
            if f.stat().st_size > 100:
                return True
    return False


def ensure_kaggle_data(
    kaggle_dir: Optional[str] = None,
    auto_download: bool = True,
) -> Optional[str]:
    """Ensure Kaggle competition data is available, downloading if necessary.

    This is the primary entry point for pipeline integration. Call this
    before accessing Kaggle data to ensure MMasseyOrdinals.csv exists.

    Respects the ``KAGGLE_NO_AUTO_DOWNLOAD`` environment variable — if set
    to ``1`` or ``true``, auto-download is disabled (useful for CI).

    To prevent repeated download attempts on every pipeline run, a sentinel
    file is written after a failed attempt.  Delete it or pass
    ``force=True`` to ``download_competition_data`` to retry.

    Args:
        kaggle_dir: Explicit path to Kaggle data directory. If None,
            searches standard locations.
        auto_download: If True, attempt to download data when not found.

    Returns:
        Path to the Kaggle data directory, or None if unavailable.
    """
    # 1. Check explicit path
    if kaggle_dir and _has_massey_ordinals(Path(kaggle_dir)):
        return kaggle_dir

    # 2. Search standard locations
    candidates = _get_kaggle_dir_candidates(kaggle_dir)
    for candidate in candidates:
        if _has_massey_ordinals(Path(candidate)):
            logger.info("Found Kaggle data at: %s", candidate)
            return candidate

    # 3. Auto-download if enabled and not suppressed
    no_auto = os.environ.get("KAGGLE_NO_AUTO_DOWNLOAD", "").lower()
    if no_auto in ("1", "true", "yes"):
        auto_download = False
        logger.info("KAGGLE_NO_AUTO_DOWNLOAD is set — skipping auto-download")

    if auto_download:
        download_dir = kaggle_dir or DEFAULT_KAGGLE_DIR
        sentinel = Path(download_dir) / _DOWNLOAD_FAILED_SENTINEL
        if sentinel.exists():
            logger.debug(
                "Previous download attempt failed (sentinel %s exists). Delete it or run with --force to retry.",
                sentinel,
            )
        else:
            logger.info(
                "Kaggle data not found locally. Attempting auto-download to %s...",
                download_dir,
            )
            result = download_competition_data(output_dir=download_dir)
            if result:
                return result
            # Write sentinel so we don't retry on next pipeline run
            try:
                Path(download_dir).mkdir(parents=True, exist_ok=True)
                sentinel.write_text("download failed\n")
            except OSError:
                pass

    logger.warning(
        "Kaggle data (MMasseyOrdinals.csv) not available. "
        "External rating composites will use seed-based fallback. "
        "To fix: set KAGGLE_USERNAME + KAGGLE_KEY env vars and run: "
        "python -m src.data.kaggle_downloader"
    )
    return None


def _get_kaggle_dir_candidates(explicit_dir: Optional[str] = None) -> List[str]:
    """Get list of directories to search for Kaggle CSVs."""
    candidates = []
    if explicit_dir:
        candidates.append(explicit_dir)

    cwd = os.getcwd()
    candidates.extend(
        [
            os.path.join(cwd, "data", "kaggle"),
            os.path.join(cwd, "data", "raw", "kaggle"),
            os.path.join(cwd, "kaggle"),
            os.path.join(cwd, "data", "raw"),
        ]
    )
    return candidates


def verify_massey_ordinals(kaggle_dir: str, season: int) -> dict:
    """Verify that Massey Ordinals are loadable for a given season.

    Returns a diagnostic dict with coverage statistics.
    """
    from .kaggle_loader import KaggleDataLoader

    loader = KaggleDataLoader(kaggle_dir)
    available_files = loader.list_available_files()
    has_massey_csv = any("massey" in f.lower() for f in available_files)
    has_teams_csv = any("teams" in f.lower() for f in available_files)

    result = {
        "kaggle_dir": kaggle_dir,
        "has_massey_csv": has_massey_csv,
        "has_teams_csv": has_teams_csv,
        "available_files": available_files,
        "season": season,
        "ordinal_systems": 0,
        "teams_covered": 0,
        "status": "not_available",
    }

    if not has_massey_csv:
        result["status"] = "missing_csv"
        return result

    ordinals = loader.load_massey_ordinals(season)
    if not ordinals:
        result["status"] = "no_data_for_season"
        return result

    n_systems = len(ordinals)
    n_teams = max((len(v) for v in ordinals.values()), default=0)
    result["ordinal_systems"] = n_systems
    result["teams_covered"] = n_teams
    result["system_names"] = sorted(ordinals.keys())
    result["status"] = "ok" if n_systems >= 5 and n_teams >= 50 else "partial"

    return result


# CLI entry point
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download Kaggle March Mania competition data (MMasseyOrdinals, etc.)")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_KAGGLE_DIR,
        help=f"Output directory for CSV files (default: {DEFAULT_KAGGLE_DIR})",
    )
    parser.add_argument(
        "--competition",
        default=None,
        help="Specific Kaggle competition slug",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if data already exists",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=None,
        metavar="SEASON",
        help="Verify Massey Ordinals for a specific season after download",
    )
    args = parser.parse_args()

    result = download_competition_data(
        output_dir=args.output_dir,
        competition=args.competition,
        force=args.force,
    )

    if result:
        print(f"Kaggle data available at: {result}")
        if args.verify:
            diag = verify_massey_ordinals(result, args.verify)
            print(f"Massey Ordinals verification for {args.verify}:")
            for k, v in diag.items():
                if k != "system_names":
                    print(f"  {k}: {v}")
                else:
                    print(f"  systems ({len(v)}): {', '.join(v[:10])}...")
            sys.exit(0 if diag["status"] in ("ok", "partial") else 1)
        sys.exit(0)
    else:
        print("Failed to download Kaggle data. See log for details.")
        sys.exit(1)
