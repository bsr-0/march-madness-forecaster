"""Prospective-integrity gate for the shipped site.

Months of methodological discipline can be undone by one convenient badge. The
easiest way to lose it is a future change adding "2026 accuracy: 10.47%" from an
old JSON field, or surfacing an algorithm name because it was the handiest label
available. This test makes those regressions fail rather than ship.

Two enforcement zones:

  PRODUCT SURFACE  docs/build.js, docs/selection.js, and the Build section of
                   docs/index.html. Strictly enforced.

  LEGACY SURFACE   the older Bracket Picker tab. It genuinely contains
                   prohibited content today, recorded below in _LEGACY_DEBT with
                   what is wrong and why. The allowlist is the point: the debt is
                   visible and enumerated rather than silently tolerated, and
                   retiring the tab means deleting entries here.

Adding to _LEGACY_DEBT is a deliberate act requiring justification, not a way to
make this test pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

BUILD_JS = DOCS / "build.js"
SELECTION_JS = DOCS / "selection.js"
INDEX = DOCS / "index.html"

# Internal vocabulary that must never reach a user. The product explains what a
# strategy does, not how the machinery works.
_INTERNAL_TERMS = [
    "Pool Optimizer",
    "Region Beam Search",
    "Exhaustive Search",
    "Monte Carlo",
    "Hamming",
    "log5",
    "P(1st)",
    "barthag",
    "scenario bank",
    "marginal",
    "pairwise",
]

# Known prohibited content in the legacy Bracket Picker tab, to be removed when
# that tab is retired (Phase 4). Enumerated so it cannot grow unnoticed.
_LEGACY_DEBT = {
    "algorithm names as strategy labels": (
        "app.js STRATEGIES and index.html expose 'Pool Optimizer', 'Region Beam "
        "Search' and 'Exhaustive Search' -- construction methods, not user goals."
    ),
    "hardcoded p_first badges": (
        "app.js carries p_first: 11.2 / 6.2 / 6.3. The 11.2 figure is STALE (the "
        "corrected value is 10.47) and, more importantly, every one of them is a "
        "backtest figure over a window that includes 2026. Patching the number "
        "would make a prohibited claim more accurate rather than removing it; the "
        "fix is retiring the tab."
    ),
    "P(1st) shown as user copy": "index.html and app.js render raw P(1st) percentages.",
}


def _user_visible_text(path: Path) -> str:
    """Text a user could actually read, not identifiers or developer comments.

    Scanning raw source produces false positives: `log5` is a function reference
    passed into the mirror, and "Balanced"/"Contrarian" appear in comments
    explaining why they are deliberately absent. Neither reaches a user. So JS is
    reduced to its string literals, and HTML to its text content.
    """
    text = path.read_text()
    if path.suffix == ".js":
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        text = re.sub(r"^\s*//.*$", " ", text, flags=re.M)
        literals = re.findall(r"'([^'\n]*)'|\"([^\"\n]*)\"|`([^`]*)`", text)
        return " ".join(x for group in literals for x in group if x)
    body = re.sub(r"<script.*?</script>", " ", text, flags=re.S)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.S)
    return re.sub(r"<[^>]+>", " ", body)


def _build_section(html: str) -> str:
    """Just the Build tab markup, so legacy content is not scanned as product."""
    start = html.find('id="tab-build"')
    end = html.find('id="tab-bracket"')
    assert start != -1 and end > start, "Build section not found in index.html"
    return html[start:end]


# ---------------------------------------------------------------------------
# Product surface — strict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term", _INTERNAL_TERMS)
def test_product_surface_has_no_internal_vocabulary(term):
    """The user chooses a goal; they never meet the machinery."""
    product_text = " ".join([
        _user_visible_text(BUILD_JS),
        re.sub(r"<[^>]+>", " ", _build_section(INDEX.read_text())),
    ])
    offenders = [term] if term.lower() in product_text.lower() else []
    assert not offenders, (
        f"internal term {term!r} appears in user-visible product copy. "
        "Describe what the strategy does for the user instead."
    )


def test_no_2026_performance_claim_in_the_product_surface():
    """2026 is an in-sample replay. Its accuracy is not evidence of anything."""
    text = _user_visible_text(BUILD_JS) + re.sub(r"<[^>]+>", " ", _build_section(INDEX.read_text()))
    banned = [
        r"\d+(\.\d+)?\s*%\s*(accuracy|correct|P\(1st\))",
        r"accuracy[:\s]+\d",
        r"beat\w*\s+the\s+(seed|market)",
        r"\d+\s*/\s*15\s+years",
    ]
    hits = [p for p in banned if re.search(p, text, re.I)]
    assert not hits, (
        f"the product surface makes a performance claim ({hits}). 2026 is a replay; "
        "its results must not be presented as predictive evidence."
    )


def test_frequencies_never_derived_from_candidate_counts():
    """The candidate list is not a probability sample.

    The sampler deliberately over-samples unlikely champions to protect
    diversity, so counting rows would bias every 'happens in N of 10
    tournaments' toward rare scenarios. Frequencies must come from the
    artifact's full-bank fields via constraintFrequency().
    """
    src = BUILD_JS.read_text() + SELECTION_JS.read_text()
    counting = [
        r"candidates\.filter\([^)]*\)\.length\s*/\s*\w*\.?candidates\.length",
        r"surviving\.length\s*/\s*\w*candidates\.length",
        r"/\s*ARTIFACT\.candidates\.length",
    ]
    hits = [p for p in counting if re.search(p, src)]
    assert not hits, (
        f"a user-facing frequency looks like it is derived by counting candidates ({hits}). "
        "Use constraintFrequency(), which reads the full-bank fields."
    )
    assert "constraintFrequency" in BUILD_JS.read_text(), (
        "build.js no longer calls constraintFrequency -- frequencies may have been "
        "reimplemented against the candidate list."
    )


def test_pool_size_assumption_is_disclosed_in_the_ui():
    """P(1st) is conditional on an opponent field; the user must be told."""
    text = BUILD_JS.read_text()
    assert "not a universal probability" in text.lower()
    assert re.search(r"\{?m?\.?p1_pool_size|30-opponent|opponent pool", text), (
        "the pool-size assumption is not surfaced alongside pool upside"
    )


def test_unvalidated_strategies_are_absent():
    """Balanced and Contrarian were never defined or measured."""
    text = _user_visible_text(BUILD_JS) + re.sub(r"<[^>]+>", " ", _build_section(INDEX.read_text()))
    for banned in ("Balanced", "Contrarian"):
        assert banned not in text, (
            f"{banned!r} appears in the product surface. It has no measured definition; "
            "adding it requires research first, not a label."
        )


def test_product_surface_offers_exactly_the_frozen_objectives():
    text = BUILD_JS.read_text()
    assert "Trust the Model" in text and "Win My Pool" in text
    assert re.search(r"key:\s*'ev'", text) and re.search(r"key:\s*'p1'", text)
    extra = set(re.findall(r"\{\s*key:\s*'(\w+)'", text)) - {"ev", "p1"}
    assert not extra, f"unexpected objective(s) in the Build flow: {extra}"


def test_replay_framing_is_present():
    """The user should know 2026 is a replay, not a live prediction."""
    assert "replay" in BUILD_JS.read_text().lower()


# ---------------------------------------------------------------------------
# Legacy surface — enumerated, not tolerated
# ---------------------------------------------------------------------------


def test_legacy_debt_is_still_accurately_described():
    """If the legacy tab is cleaned up, delete the entry rather than leaving it.

    This fails when _LEGACY_DEBT claims a problem that no longer exists, so the
    allowlist cannot rot into a permanent excuse.
    """
    app = (DOCS / "app.js").read_text()
    index = INDEX.read_text()
    still_present = {
        "algorithm names as strategy labels": "Pool Optimizer" in app or "Pool Optimizer" in index,
        "hardcoded p_first badges": "p_first: 11.2" in app,
        "P(1st) shown as user copy": "P(1st)" in index,
    }
    stale = [k for k, present in still_present.items() if not present]
    assert not stale, f"_LEGACY_DEBT lists problems that are now fixed: {stale}. Remove the entries."


def test_legacy_debt_has_not_grown():
    """New prohibited content must not be added to the legacy tab either."""
    assert len(_LEGACY_DEBT) == 3, (
        "the legacy allowlist changed size. Growing it means new prohibited content "
        "shipped; shrinking it means an entry should have been deleted."
    )


def test_stale_production_figure_is_recorded_as_debt():
    """11.2% is stale (10.47%) AND drawn from a 2026-inclusive window.

    Recorded rather than patched: correcting the digits would make a prohibited
    claim more accurate instead of removing it.
    """
    assert "10.47" in _LEGACY_DEBT["hardcoded p_first badges"]
    assert "includes 2026" in _LEGACY_DEBT["hardcoded p_first badges"]
