"""Prospective-integrity gate for the shipped site.

Months of methodological discipline can be undone by one convenient badge. The
easiest way to lose it is a future change adding "2026 accuracy: 10.47%" from an
old JSON field, or surfacing an algorithm name because it was the handiest label
available. This test makes those regressions fail rather than ship.

There is now ONE enforcement zone. Every script the browser loads is scanned as
product surface; the "legacy tab" exemption is gone because the legacy tab is
gone. _LEGACY_DEBT is empty and must stay that way — entries are retired by
deleting code, never by widening the allowlist.

The sharpest rule here concerns the replay year. 2026 is in-sample (spec 2027.v2
trains through it) and is also the model's best season, so including it flatters
every figure. That exclusion is enforced in Python, and this file checks both
that the payload excludes it and that the browser does not rebuild a
contaminated headline from per-year rows.
"""

from __future__ import annotations

import json
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

# The legacy allowlist is now EMPTY, and that is the point: every entry was
# retired by deleting the code, not by widening the exemption.
#
#   algorithm names as strategy labels -- the STRATEGIES catalog naming
#       "Pool Optimizer" / "Region Beam Search" / "Exhaustive Search" is deleted.
#   hardcoded p_first badges -- the 11.2 / 6.2 / 6.3 figures are deleted. They
#       were drawn from a window that included the replay year, so patching the
#       digits would have made a prohibited claim more accurate rather than
#       removing it.
#   P(1st) shown as user copy -- the panels that rendered it are gone.
#
# Adding an entry here is a deliberate act requiring justification, not a way to
# make this file pass.
_LEGACY_DEBT: dict = {}

# Every script the browser actually loads. Scanned as one product surface --
# there is no longer a second-class "legacy" zone.
PRODUCT_SCRIPTS = ("build.js", "selection.js", "explore.js", "record.js", "app.js")


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
        # Thrown errors are developer diagnostics, not product copy. They abort
        # rendering rather than appearing on screen, and their whole value is
        # naming the internal thing that broke -- "artifact is missing the
        # pairwise table" is exactly the message a developer needs. Stripping
        # them keeps the scan honest about what a *user* can read.
        text = re.sub(r"throw new Error\([^;]*\);", " ", text, flags=re.S)
        # Schema field-name lists are data-structure vocabulary, not copy. The
        # contract has to name `pairwise` in order to require it; that string is
        # a key, and no user ever sees it.
        text = re.sub(r"const \w*_FIELDS\s*=\s*\[[^\]]*\];", " ", text, flags=re.S)
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
    product_text = " ".join(
        [_user_visible_text(DOCS / name) for name in PRODUCT_SCRIPTS] + [_user_visible_text(INDEX)]
    )
    offenders = [term] if term.lower() in product_text.lower() else []
    assert not offenders, (
        f"internal term {term!r} appears in user-visible product copy. "
        "Describe what the strategy does for the user instead."
    )


def test_no_2026_performance_claim_in_the_product_surface():
    """2026 is an in-sample replay. Its accuracy is not evidence of anything."""
    text = " ".join(
        [_user_visible_text(DOCS / name) for name in PRODUCT_SCRIPTS] + [_user_visible_text(INDEX)]
    )
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
    # v1 ships no preference controls, so no "happens in X of 10 tournaments"
    # figure is displayed and constraintFrequency() is legitimately uncalled. The
    # guard therefore binds conditionally: the moment the product shows a
    # frequency again, it must come through the mirror's full-bank accessor.
    build_src = BUILD_JS.read_text()
    shows_frequency = re.search(r"of 10 tournaments|in \$\{.*\} of 10|frequency", build_src, re.I)
    if shows_frequency:
        assert "constraintFrequency" in build_src, (
            "build.js displays a frequency without calling constraintFrequency -- it "
            "may have been reimplemented against the candidate list."
        )
    # Either way, the product must never read the full-bank fields itself and do
    # its own arithmetic; that is the mirror's job.
    for field in ("constraint_probabilities", "team_final_four_probabilities"):
        assert field not in build_src, (
            f"build.js reads {field} directly instead of going through "
            "constraintFrequency() in the parity-tested mirror."
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
    text = " ".join(
        [_user_visible_text(DOCS / name) for name in PRODUCT_SCRIPTS] + [_user_visible_text(INDEX)]
    )
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


def test_legacy_machinery_is_deleted_not_hidden():
    """The old code must be gone from the bundle, not merely unreferenced.

    Hiding a panel leaves the vocabulary one edit away from returning. These
    symbols were the product's second implementation of the model and its
    hardcoded performance claims.
    """
    app = (DOCS / "app.js").read_text()
    gone = {
        "client-side simulation": r"\bfunction simulate\b",
        "browser log5": r"\bfunction log5\b",
        "strategy catalog": r"\bconst STRATEGIES\b",
        "hardcoded P(1st) badges": r"\bp_first\b",
        "per-strategy winner resolution": r"\bfunction pick\s*\(",
    }
    still = [name for name, pat in gone.items() if re.search(pat, app)]
    assert not still, f"legacy machinery still present in app.js: {still}"


def test_no_algorithm_names_anywhere_in_the_bundle():
    """Construction methods are not user-facing vocabulary."""
    for name in PRODUCT_SCRIPTS:
        text = _user_visible_text(DOCS / name)
        for term in ("Pool Optimizer", "Region Beam Search", "Exhaustive Search", "Chalk"):
            assert term not in text, f"{name} exposes the algorithm name {term!r}"


def test_stale_performance_figures_are_gone():
    """11.2 / 6.2 / 6.3 came from a window that included the replay year."""
    for name in PRODUCT_SCRIPTS:
        text = _user_visible_text(DOCS / name)
        for stale in ("11.2%", "6.2%", "6.3%", "4.9%"):
            assert stale not in text, f"{name} still shows the stale figure {stale}"


def test_legacy_debt_is_empty():
    """If something must be re-added, it needs a written justification here."""
    assert _LEGACY_DEBT == {}, (
        f"legacy debt reappeared: {sorted(_LEGACY_DEBT)}. Entries are removed by "
        "deleting code, never by adding exemptions."
    )


# ---------------------------------------------------------------------------
# Track Record — the replay year may never become evidence
# ---------------------------------------------------------------------------


def test_track_record_headline_excludes_the_replay_year():
    """THE integrity gate for this page.

    2026 is in-sample (spec 2027.v2 trains through it) AND is the model's best
    season, so including it inflates every headline. The exclusion is done in
    Python before the payload is written.
    """
    payload = json.loads((DOCS / "data" / "ml_backtest.json").read_text())
    assert 2026 not in payload["years"], (
        "the replay year is back in the scored window; every headline figure on "
        "the Track Record page would then be a claim about a season the model "
        "was trained on"
    )
    assert payload["replay_year"]["year"] == 2026
    assert payload["replay_year"]["is_out_of_sample"] is False
    scored = {r["year"] for r in payload["per_year"] if r["year"] in payload["years"]}
    assert 2026 not in scored


def test_browser_never_reaggregates_a_headline_from_per_year_rows():
    """Re-aggregating would silently rebuild the contaminated number."""
    src = (DOCS / "record.js").read_text()
    assert not re.search(r"per_year[\s\S]{0,200}reduce\(", src), (
        "record.js aggregates per_year rows; headline figures must come from the "
        "payload's precomputed models block, which already excludes the replay year"
    )
    assert "all_years_including_replay" not in src, (
        "record.js reads the unfiltered year list"
    )


def test_replay_is_labelled_as_in_sample_wherever_it_appears():
    text = _user_visible_text(DOCS / "record.js")
    assert "not a result" in text or "in-sample" in text.lower()
    assert "self-test" in text.lower() or "not a prediction" in text.lower()


