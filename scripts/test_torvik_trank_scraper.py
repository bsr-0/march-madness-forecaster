#!/usr/bin/env python3
"""Quick smoke-test for Torvik trank.php CSV scraping with browser headers.

Tests multiple HTTP strategies (requests, cloudscraper, curl_cffi) against
the live barttorvik.com/trank.php endpoint and reports which ones succeed.

Usage:
    python scripts/test_torvik_trank_scraper.py [--year 2026]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time

# ---------------------------------------------------------------------------
# Browser-like headers — the key to passing Cloudflare's light verification
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://barttorvik.com/trank.php",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

BASE_URL = "https://barttorvik.com/trank.php"


def _build_params(year: int) -> dict:
    return {
        "year": year,
        "csv": 1,
        "conyes": 1,
        "type": "All",
        "top": 0,
    }


def _is_cloudflare_block(text: str) -> bool:
    snippet = text[:1000].lower()
    return "<html" in snippet and (
        "checking your browser" in snippet
        or "cloudflare" in snippet
        or "cf-browser-verification" in snippet
        or "just a moment" in snippet
    )


def _parse_csv_preview(text: str, max_rows: int = 5) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text))
    rows = []
    for i, row in enumerate(reader):
        if i >= max_rows + 1:  # +1 for header
            break
        rows.append(row)
    return rows


def _count_csv_rows(text: str) -> int:
    reader = csv.reader(io.StringIO(text))
    count = 0
    for _ in reader:
        count += 1
    return max(0, count - 1)  # subtract header


# ---------------------------------------------------------------------------
# Strategy 1: plain requests
# ---------------------------------------------------------------------------
def test_requests(year: int) -> tuple[bool, str]:
    try:
        import requests
    except ImportError:
        return False, "requests not installed"

    session = requests.Session()
    session.headers.update(HEADERS)
    params = _build_params(year)

    t0 = time.time()
    try:
        resp = session.get(BASE_URL, params=params, timeout=30)
        elapsed = time.time() - t0
    except Exception as e:
        return False, f"request failed: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} ({time.time() - t0:.1f}s)"

    if _is_cloudflare_block(resp.text):
        return False, f"Cloudflare challenge page ({elapsed:.1f}s)"

    n_teams = _count_csv_rows(resp.text)
    if n_teams < 50:
        return False, f"only {n_teams} teams parsed ({elapsed:.1f}s) — likely not real data"

    return True, f"{n_teams} teams in {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Strategy 2: cloudscraper (Cloudflare bypass)
# ---------------------------------------------------------------------------
def test_cloudscraper(year: int) -> tuple[bool, str]:
    try:
        import cloudscraper
    except ImportError:
        return False, "cloudscraper not installed"

    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "darwin", "desktop": True})
    scraper.headers.update(HEADERS)
    params = _build_params(year)

    t0 = time.time()
    try:
        resp = scraper.get(BASE_URL, params=params, timeout=30)
        elapsed = time.time() - t0
    except Exception as e:
        return False, f"request failed: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} ({elapsed:.1f}s)"

    if _is_cloudflare_block(resp.text):
        return False, f"Cloudflare challenge page ({elapsed:.1f}s)"

    n_teams = _count_csv_rows(resp.text)
    if n_teams < 50:
        return False, f"only {n_teams} teams parsed ({elapsed:.1f}s) — likely not real data"

    return True, f"{n_teams} teams in {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Strategy 3: curl_cffi (Chrome TLS fingerprint impersonation)
# ---------------------------------------------------------------------------
def test_curl_cffi(year: int) -> tuple[bool, str]:
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return False, "curl_cffi not installed"

    params = _build_params(year)

    t0 = time.time()
    try:
        resp = curl_requests.get(
            BASE_URL,
            params=params,
            headers=HEADERS,
            impersonate="chrome",
            timeout=30,
        )
        elapsed = time.time() - t0
    except Exception as e:
        return False, f"request failed: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} ({elapsed:.1f}s)"

    if _is_cloudflare_block(resp.text):
        return False, f"Cloudflare challenge page ({elapsed:.1f}s)"

    n_teams = _count_csv_rows(resp.text)
    if n_teams < 50:
        return False, f"only {n_teams} teams parsed ({elapsed:.1f}s) — likely not real data"

    return True, f"{n_teams} teams in {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Strategy 4: curl_cffi with Session (reusable connections)
# ---------------------------------------------------------------------------
def test_curl_cffi_session(year: int) -> tuple[bool, str]:
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return False, "curl_cffi not installed"

    params = _build_params(year)

    t0 = time.time()
    try:
        session = curl_requests.Session(impersonate="chrome")
        session.headers.update(HEADERS)
        resp = session.get(BASE_URL, params=params, timeout=30)
        elapsed = time.time() - t0
    except Exception as e:
        return False, f"request failed: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} ({elapsed:.1f}s)"

    if _is_cloudflare_block(resp.text):
        return False, f"Cloudflare challenge page ({elapsed:.1f}s)"

    n_teams = _count_csv_rows(resp.text)
    if n_teams < 50:
        return False, f"only {n_teams} teams parsed ({elapsed:.1f}s) — likely not real data"

    # Show a preview of the data
    rows = _parse_csv_preview(resp.text, max_rows=3)
    preview = "\n".join("  | ".join(r[:6]) for r in rows[:4])

    return True, f"{n_teams} teams in {elapsed:.1f}s\n  Preview:\n  {preview}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STRATEGIES = [
    ("requests (plain)", test_requests),
    ("cloudscraper", test_cloudscraper),
    ("curl_cffi (one-shot)", test_curl_cffi),
    ("curl_cffi (session)", test_curl_cffi_session),
]


def main():
    parser = argparse.ArgumentParser(description="Test Torvik trank.php scraper strategies")
    parser.add_argument("--year", type=int, default=2026, help="Season year (default: 2026)")
    args = parser.parse_args()

    print(f"Testing trank.php scraper — year={args.year}")
    print(f"URL: {BASE_URL}?year={args.year}&csv=1&conyes=1&type=All&top=0")
    print("=" * 70)

    results = {}
    for name, fn in STRATEGIES:
        print(f"\n[{name}] ...", flush=True)
        ok, msg = fn(args.year)
        status = "PASS" if ok else "FAIL"
        results[name] = (ok, msg)
        print(f"[{name}] {status}: {msg}")
        # Small delay between strategies to avoid rate limiting
        time.sleep(1)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    any_pass = False
    for name, (ok, msg) in results.items():
        icon = "+" if ok else "-"
        short_msg = msg.split("\n")[0]
        print(f"  [{icon}] {name}: {short_msg}")
        if ok:
            any_pass = True

    if any_pass:
        print("\nAt least one strategy succeeded.")
    else:
        print("\nAll strategies failed — Cloudflare may be blocking all requests.")

    return 0 if any_pass else 1


if __name__ == "__main__":
    sys.exit(main())
