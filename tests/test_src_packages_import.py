"""Every package under src/ must import.

Commit 44b048f (2026-04-21, "upd") deleted 124 files under src/ and left 37
imports pointing at them. Most were lazy or guarded and only failed when an
optional feature ran. Two were package ``__init__`` files eagerly importing
deleted submodules, which made the *surviving* modules in those packages
unimportable too -- `src.ml.training.symmetric` (used by every training run)
and `src.espn.public_pick_scraper`. Together with a module-level import in
`data_loader`, that is how the ML pipeline went five months without being able
to load data, unnoticed, because the harness substituted the seed baseline.

This test imports each package ``__init__`` and fails on the first
ModuleNotFoundError. It is deliberately packages-only: leaf modules with
optional heavy dependencies (torch, lightgbm) are someone else's concern.
"""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Enumerate from the filesystem, not pkgutil.walk_packages: the latter has to
# import a package to descend into it, so one broken __init__ silently hides
# every package beneath it -- which is the failure mode being tested for.
PACKAGES = sorted(
    ".".join(p.parent.relative_to(ROOT).parts)
    for p in (ROOT / "src").rglob("__init__.py")
    if "__pycache__" not in p.parts
)


@pytest.mark.parametrize("package", PACKAGES)
def test_package_init_imports(package):
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as exc:
        if str(exc).startswith("No module named 'src"):
            pytest.fail(f"{package}: {exc} -- a src module is imported that no longer exists")
        pytest.skip(f"{package}: optional third-party dependency missing ({exc})")
