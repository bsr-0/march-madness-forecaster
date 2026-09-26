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
    assert page.locator("#empty.wait").inner_text().startswith("2027 field not announced yet")
    assert top(page, "#empty") < top(page, "#compare") < top(page, "#rulepanel") < top(page, "#board")
    assert page.locator("#compare .scard").count() == 5
    assert page.locator("#compare .scard.na").count() == 4
    assert page.locator("#rule-body .rule-row").count() == 3
    assert page.locator("#board > .round").count() == 6
    counts = page.eval_on_selector_all("#board .r-count", "els => els.map(e => e.textContent)")
    assert counts == ["32 games", "16 games", "8 games", "4 games", "2 games", "1 game"]
    assert page.locator("#board .game").count() == 0
    assert page.locator("#board .r-sub").count() == 6
    for hidden in ("#headline", "#tune", "#explore", "#why", "#board-tools"):
        assert not visible(page, hidden), hidden


def test_missing_recommended_payload_does_not_render_p1_under_its_name(site, page):
    page.goto(site + "#y=2026&s=recommended")
    page.wait_for_function("state.season && state.season.status === 'ready'", timeout=SEARCH_TIMEOUT)
    page.evaluate("""
      state.season.strategies = state.season.strategies.filter(s => s.id !== 'recommended');
      state.season.year = 2027;
      state.year = 2027;
      state.strategy = 'recommended';
      render();
    """)
    assert page.evaluate("""() => {
      const rows = strategyRows();
      const rec = rows.find(row => row.id === RECOMMENDED);
      const p1 = rows.find(row => row.id === 'p1');
      return !currentStrategy()
        && !state.rounds
        && rec && rec.unavailable && rec.pending && !rec.active
        && p1 && !p1.active;
    }""")
    assert "No verified Recommended bracket is present" in page.locator("#compare").text_content()
    assert page.locator("#board > .round").count() == 0
    assert page.locator("#compare .scard[data-strategy='recommended'].na").count() == 1
    assert page.locator("#compare .scard[data-strategy='p1']").get_attribute("aria-pressed") == "false"


def test_a_chosen_rule_travels_in_rq_and_is_said_when_2026_lacks_it(site, page):
    # No Simple rule reproduces both 2026 and 2025, so a rule chosen on 2027
    # (run: 2026) is not a survivor on 2026 (run: 2025): the page says so and
    # keeps the link's rule rather than quietly showing another under its name.
    page.goto(site)
    settle(page)
    page.locator("#rule-body .rule-row").nth(1).click()
    assert "rq=" in page.evaluate("location.hash")
    want = page.evaluate("state.rule.want")
    page.click(".yr[data-year='2026']")
    settle(page)
    assert page.evaluate("state.rule.result.wantMissed") is True
    assert page.locator("#rule-body .ex-line:has-text('The rule this link names')").is_visible()
    assert page.locator("#board .side.picked").count() == 63
    assert page.evaluate("state.rule.want") == want
    assert "rq=" in page.evaluate("location.hash")
    labels = page.eval_on_selector_all("#board .r-sub", "els => els.map(e => e.textContent)")
    assert labels == page.evaluate("ruleChosen().seq.map(ruleLabel)")


def test_one_variable_table_is_collapsed_after_the_alternatives(site, page):
    page.goto(site + "#y=2026&s=rule")
    settle(page)
    table = page.locator("#rule-body details.one-more")
    assert table.count() == 1
    assert page.evaluate("document.querySelector('#rule-body details.one-more').open") is False
    assert top(page, "#rule-body .rule-row") < top(page, "#rule-body details.one-more")
    assert page.locator("#rule-body select").count() == 0, "no pickers left on the panel"
    table.locator("summary").click()
    assert page.locator("#rule-body .one-tbl tbody tr").count() >= 30


def test_filters_gate_per_strategy_on_2026(site, page):
    page.goto(site + "#y=2026")
    page.wait_for_function("() => document.querySelectorAll('#board .side.picked').length === 63")
    page.locator("#customize > summary").click()
    page.click("#compare .scard[data-strategy='p1']")
    assert visible(page, "#tune") and not visible(page, "#explore") and not visible(page, "#rulepanel")
    page.click("#compare .scard[data-strategy='model']")
    page.wait_for_function("() => !document.getElementById('explore').hidden")
    assert not visible(page, "#tune") and not visible(page, "#rulepanel")
    page.click("#compare .scard[data-strategy='rule']")
    settle(page)
    assert visible(page, "#rulepanel") and not visible(page, "#tune") and not visible(page, "#explore")


def test_pool_variant_selector_round_trips_and_changes_metrics(site, page):
    page.goto(site + "#y=2026&pool=50")
    page.wait_for_function("() => document.querySelectorAll('#board .side.picked').length === 63")
    assert page.evaluate("state.poolSize") == 50
    page.click("#customize > summary")
    assert page.locator("#pool-size").input_value() == "50"
    p1_50 = page.evaluate("strategyRows().find(x => x.id === 'p1').p1")
    page.select_option("#pool-size", "10")
    page.wait_for_function("() => state.poolSize === 10 && document.querySelectorAll('#board .side.picked').length === 63")
    assert "pool=10" in page.evaluate("location.hash")
    p1_10 = page.evaluate("strategyRows().find(x => x.id === 'p1').p1")
    assert p1_10 != p1_50


def test_checkpoint_guard_shows_its_message(site, page):
    page.goto(site + "#y=2026&s=rule")
    settle(page)
    assert page.evaluate("state.rule.checkpoints") == [3, 5]
    assert page.locator("#rule-body .rule-line .chip").count() == 6
    page.click("#rule-body .chip:has-text('Final Four')")   # would leave the champion alone: refused
    assert page.locator("#rule-body .rule-hint.warn").is_visible()
    assert page.locator("#rule-body .rule-hint.warn").inner_text().startswith("Keep at least one of Sweet 16")
    assert page.evaluate("state.rule.checkpoints") == [3, 5]


def test_a_skipped_newest_season_is_said_and_the_run_starts_below_it(site, page):
    # 2024 has no one-variable-per-round survivor for the Final Four at any
    # cap: on 2025 the search says so and starts the run at the most recent
    # season a rule reproduces.
    page.goto(site + "#y=2025&s=rule")
    settle(page)
    line = page.locator("#rule-body .ex-line:has-text('No simple rule reproduces')")
    assert line.inner_text().startswith("No simple rule reproduces final four and champion in ")
    assert line.locator(".chip:has-text('Try Flexible')").count() == 1
    skipped = page.evaluate("state.rule.result.skipped")
    assert skipped[0] == 2024
    assert page.evaluate("state.rule.result.run[0]") < 2024
    assert page.locator("#board .side.picked").count() == 63
    assert visible(page, "#compare")
    line.locator(".chip:has-text('Try Flexible')").click()
    settle(page)
    assert page.evaluate("state.rule.maxCriteria") == 3
    assert page.evaluate("state.rule.result.skipped")[0] == 2024


def test_search_runs_off_the_main_thread_and_the_latest_controls_win(site, page):
    page.goto(site + "#y=2026&s=rule")
    settle(page)
    page.evaluate("window.__ticks = 0; (function loop() { window.__ticks++; requestAnimationFrame(loop); })();")
    page.click("#rule-body .chip:has-text('Flexible')")    # the heavier of the two settings
    page.wait_for_function("() => state.rule.busy")
    assert "Searching" in page.locator("#rule-body").inner_text(), "the busy state is painted"
    assert page.evaluate("state.rule.job !== null"), "the search is in a worker"
    ticks0 = page.evaluate("window.__ticks")
    page.wait_for_timeout(1000)
    if page.evaluate("state.rule.busy"):
        assert page.evaluate("window.__ticks") - ticks0 >= 20, "the page keeps painting during the search"
    page.evaluate("setRuleComplexity('simple')")              # a control change mid-search
    settle(page)
    assert page.evaluate("state.rule.maxCriteria") == 2
    assert page.evaluate("state.rule.job === null")
    assert page.evaluate("state.rule.error") is None
