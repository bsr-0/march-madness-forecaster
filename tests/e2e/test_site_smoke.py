"""Browser smoke test over docs/ -- the rendered page, driven in Chromium.

The node tests in tests/test_*.js drive app.js through DOM stubs, which is
why two rendering bugs (hand mode never redrawing; the pending table's rows
losing their cells, 2026-09-17) passed them and were caught only by hand in
a browser. This is that hand check, kept: the page served from docs/ exactly
as GitHub Pages serves it, the states a visitor actually reaches, and no
page error anywhere. Skipped where Playwright is not installed; CI installs
it (job ``site-smoke``).

Run locally:  python3 -m pytest -p no:asyncio -o addopts= tests/e2e -m e2e -v
"""

from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
SETTLED = "() => state.rule.result && !state.rule.busy && state.rule.result.key === ruleKey()"
SEARCH_TIMEOUT = 120_000   # ms; the search takes 5-15 s on a CI runner


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_args):  # noqa: D102 - silence per-request logging
        pass


@pytest.fixture(scope="module")
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(DOCS)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, request):
    """A fresh context per test; any page error or console error fails it."""
    viewport = getattr(request, "param", (1280, 900))
    ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    yield pg
    ctx.close()
    assert errors == [], errors


def settle(pg):
    pg.wait_for_function(SETTLED, timeout=SEARCH_TIMEOUT)


def top(pg, selector):
    return pg.evaluate(f"document.querySelector('{selector}').getBoundingClientRect().top")


def visible(pg, selector):
    return pg.evaluate(f"(() => {{ const e = document.querySelector('{selector}'); return !!e && !e.hidden; }})()")


@pytest.mark.parametrize("page", [(1280, 900), (390, 844)], ids=["desktop", "phone"], indirect=True)
def test_landing_opens_pending_with_orientation_first(site, page):
    page.goto(site)
    page.wait_for_selector("#main.pending")
    settle(page)
    assert "y=2027" in page.evaluate("location.hash") and "s=rule" in page.evaluate("location.hash")
    assert visible(page, "#empty.wait")
    assert top(page, "#empty") < top(page, "#compare") < top(page, "#rulepanel") < top(page, "#board")
    assert page.locator("#compare .cmp-row").count() == 4
    assert page.locator("#compare .cmp-row.na").count() == 3
    assert page.locator("#rule-body .rule-row").count() >= 1
    assert page.locator("#board > .round").count() == 6
    counts = page.eval_on_selector_all("#board .r-count", "els => els.map(e => e.textContent)")
    assert counts == ["32 games", "16 games", "8 games", "4 games", "2 games", "1 game"]
    assert page.locator("#board .game").count() == 0
    assert page.locator("#board .r-sub").count() == 6
    for hidden in ("#headline", "#tune", "#explore", "#why", "#board-tools"):
        assert not visible(page, hidden), hidden


def test_choosing_a_rule_writes_rq_and_fills_2026(site, page):
    page.goto(site)
    settle(page)
    rows = page.locator("#rule-body .rule-row")
    rows.nth(1 if rows.count() > 1 else 0).click()
    assert "rq=" in page.evaluate("location.hash")
    want = page.evaluate("state.rule.want")
    page.click(".yr[data-year='2026']")
    settle(page)
    assert page.locator("#board .side.picked").count() == 63
    assert page.evaluate("ruleChosen().seq") == want
    labels = page.eval_on_selector_all("#board .r-sub", "els => els.map(e => e.textContent)")
    assert labels == page.evaluate("state.rule.want.map(ruleLabel)")


def test_hand_mode_renders_its_result_line(site, page):
    page.goto(site + "#y=2026&s=rule")
    settle(page)
    page.click("#rule-body details.rule-more:not(.one-more) > summary")   # 'More', not the variables table
    page.click("#rule-body .chip:has-text('compose by hand')")
    settle(page)
    line = page.locator("#rule-body .ex-line:has-text('This rule')").inner_text()
    assert line.startswith("This rule gives 2026 a bracket with champion"), line
    assert page.locator("#board .side.picked").count() == 63
    page.click(".yr[data-year='2027']")
    settle(page)
    line = page.locator("#rule-body .ex-line:has-text('This rule')").inner_text()
    assert line.startswith("This rule fills the 2027 bracket once the field is out"), line


def test_filters_gate_per_strategy_on_2026(site, page):
    page.goto(site + "#y=2026")
    page.wait_for_function("() => document.querySelectorAll('#board .side.picked').length === 63")
    page.click("#compare .cmp-row:has-text('Win the pool')")
    assert visible(page, "#tune") and not visible(page, "#explore") and not visible(page, "#rulepanel")
    page.click("#compare .cmp-row:has-text('Fitted model')")
    page.wait_for_function("() => !document.getElementById('explore').hidden")
    assert not visible(page, "#tune") and not visible(page, "#rulepanel")
    page.click("#compare .cmp-row:has-text('Rule search')")
    settle(page)
    assert visible(page, "#rulepanel") and not visible(page, "#tune") and not visible(page, "#explore")


def test_checkpoint_guard_shows_its_message(site, page):
    page.goto(site + "#y=2026&s=rule")
    settle(page)
    assert page.evaluate("state.rule.checkpoints") == [3, 4, 5]
    page.click("#rule-body .chip:has-text('Final Four')")   # would leave finalists + champion: refused
    assert page.locator("#rule-body .rule-hint.warn").is_visible()
    assert page.locator("#rule-body .rule-hint.warn").inner_text().startswith("Keep at least one of Round of 32")
    assert page.evaluate("state.rule.checkpoints") == [3, 4, 5]


def test_search_runs_off_the_main_thread_and_the_latest_controls_win(site, page):
    page.goto(site + "#y=2026&s=rule")
    settle(page)
    page.evaluate("window.__ticks = 0; (function loop() { window.__ticks++; requestAnimationFrame(loop); })();")
    page.click("#rule-body .chip:has-text('last 3')")     # 2023-2025: the slow case
    page.wait_for_function("() => state.rule.busy")
    assert "Searching" in page.locator("#rule-body").inner_text(), "the busy state is painted"
    assert page.evaluate("state.rule.job !== null"), "the search is in a worker"
    ticks0 = page.evaluate("window.__ticks")
    page.wait_for_timeout(1000)
    if page.evaluate("state.rule.busy"):
        assert page.evaluate("window.__ticks") - ticks0 >= 20, "the page keeps painting during the search"
    page.evaluate("setRuleRank('outside')")                 # a control change mid-search
    settle(page)
    assert page.evaluate("state.rule.rank") == "outside"
    assert page.evaluate("state.rule.job === null")
    assert page.evaluate("state.rule.error") is None
