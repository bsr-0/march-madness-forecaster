"""CLI subcommands for scraping data."""

from pathlib import Path


def scrape_rosters(args):
    """Scrape cbbpy box scores to build per-season roster payloads."""
    import time
    from ..data.scrapers.cbbpy_rosters import CBBpyRosterScraper

    start = args.start_year
    end = args.end_year
    cache_dir = args.cache_dir
    scraper = CBBpyRosterScraper(cache_dir=cache_dir)
    delay = args.delay
    succeeded = []
    failed = []
    for year in range(start, end + 1):
        cache_path = Path(cache_dir) / f"cbbpy_rosters_{year}.json"
        if not args.force and cache_path.exists():
            print(f"[{year}] cached — skipping (use --force to re-scrape)")
            succeeded.append(year)
            continue
        print(f"[{year}] scraping season box scores via cbbpy ...")
        try:
            payload = scraper.fetch_rosters(year)
            n_teams = len(payload.get("teams", []))
            n_players = sum(len(t.get("players", [])) for t in payload.get("teams", []))
            if n_teams == 0:
                print(f"[{year}] WARNING: got 0 teams — ESPN may not have data this far back")
                failed.append(year)
            else:
                print(f"[{year}] OK — {n_teams} teams, {n_players} players")
                succeeded.append(year)
        except Exception as exc:
            print(f"[{year}] FAILED — {exc}")
            failed.append(year)
        if year < end and delay > 0:
            time.sleep(delay)
    print(f"\nDone. Succeeded: {len(succeeded)}, Failed: {len(failed)}")
    if failed:
        print(f"Failed years: {failed}")
    print(f"Cache dir: {cache_dir}")
    return 0 if not failed else 1


def enrich_rosters(args):
    """Cross-reference cbbpy rosters across years to populate eligibility_year and is_transfer."""
    from ..data.scrapers.roster_enrichment import RosterEnrichment
    enricher = RosterEnrichment(roster_dir=args.roster_dir, output_dir=args.output_dir)
    summary = enricher.enrich_all(start_year=args.start_year, end_year=args.end_year)
    print(f"\nEnriched {summary['total_players_enriched']} players across {summary['years_processed']} years")
    print(f"Transfers detected: {summary['total_transfers']}")
    print(f"Eligibility distribution: {summary['eligibility_distribution']}")
    return 0


def scrape_tournament_results(args):
    """Scrape tournament game results from Sports Reference."""
    from ..data.scrapers.tournament_results import TournamentResultsScraper
    from ..data.historical_tournament_results import (
        TournamentGame,
        save_tournament_results,
    )

    if args.year:
        years = [args.year]
    elif args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
    else:
        years = [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]

    scraper = TournamentResultsScraper(cache_dir=args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        try:
            games = scraper.scrape_year(year)
            if not games:
                print(f"  {year}: No games scraped")
                continue

            # Convert to TournamentGame objects and save
            tg_list = [
                TournamentGame(
                    year=g["year"],
                    round_name=g["round_name"],
                    team1_id=g["team1_id"],
                    team2_id=g["team2_id"],
                    team1_seed=g["team1_seed"],
                    team2_seed=g["team2_seed"],
                    team1_score=g.get("team1_score", 0),
                    team2_score=g.get("team2_score", 0),
                    team1_won=g["team1_won"],
                    region=g.get("region", ""),
                )
                for g in games
            ]
            save_tournament_results(tg_list, year, str(output_dir))
            print(f"  {year}: {len(tg_list)} games saved")
        except Exception as e:
            print(f"  {year}: Failed - {e}")

    return 0


def run_scrape_conference_seeds(args):
    """Scrape conference tournament seeds from ESPN."""
    from ..data.scrapers.conference_seeds import ConferenceSeedsScraper

    output = args.output or f"data/raw/seed_overrides_{args.year}.json"
    scraper = ConferenceSeedsScraper(cache_dir=args.cache_dir)

    print(f"Scraping conference tournament seeds for {args.year}...")
    seeds = scraper.scrape_seeds(args.year, conferences=args.conferences)

    if not seeds:
        print("No seeds scraped. ESPN API may be unavailable.")
        return 1

    scraper.save_to_file(seeds, output)
    print(f"\nScraped seeds for {len(seeds)} conferences:")
    for conf, team_seeds in sorted(seeds.items()):
        print(f"  {conf}: {len(team_seeds)} teams")
    print(f"\nSaved to {output}")
    return 0


def register(subparsers):
    """Register all scraping subcommands."""

    # --- scrape-rosters ---
    roster_parser = subparsers.add_parser(
        "scrape-rosters",
        help="Scrape cbbpy box scores to build per-season roster payloads (resumable, cached)",
    )
    roster_parser.add_argument(
        "--start-year", type=int, default=2005, help="First season to scrape (inclusive, default: 2005)"
    )
    roster_parser.add_argument(
        "--end-year", type=int, default=2026, help="Last season to scrape (inclusive, default: 2026)"
    )
    roster_parser.add_argument(
        "--cache-dir",
        default="data/raw/historical",
        help="Directory to cache roster JSONs (default: data/raw/historical)",
    )
    roster_parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds to wait between seasons (default: 2.0)"
    )
    roster_parser.add_argument(
        "--force", action="store_true", help="Re-scrape even if cached file exists"
    )
    roster_parser.set_defaults(func=scrape_rosters)

    # --- enrich-rosters ---
    enrich_parser = subparsers.add_parser(
        "enrich-rosters",
        help="Cross-reference cbbpy rosters across years to populate eligibility_year and is_transfer",
    )
    enrich_parser.add_argument(
        "--roster-dir",
        default="data/raw/historical",
        help="Directory containing cbbpy_rosters_{year}.json files (default: data/raw/historical)",
    )
    enrich_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for enriched files (default: same as roster-dir, in-place update)",
    )
    enrich_parser.add_argument(
        "--start-year", type=int, default=2005,
        help="First season to process (inclusive, default: 2005)",
    )
    enrich_parser.add_argument(
        "--end-year", type=int, default=2026,
        help="Last season to process (inclusive, default: 2026)",
    )
    enrich_parser.set_defaults(func=enrich_rosters)

    # --- scrape-tournament-results ---
    str_parser = subparsers.add_parser(
        "scrape-tournament-results",
        help="Scrape historical tournament game results from Sports Reference",
    )
    str_parser.add_argument("--year", type=int, default=None, help="Single year to scrape")
    str_parser.add_argument("--years", default=None, help="Comma-separated years (default: 2018-2019,2021-2025)")
    str_parser.add_argument("--cache-dir", default="data/raw/cache", help="Cache directory for HTTP responses")
    str_parser.add_argument("--output-dir", default="data/raw/historical", help="Output directory for tournament_results_YYYY.json")
    str_parser.add_argument("--delay", type=float, default=3.0, help="Seconds between requests")
    str_parser.set_defaults(func=scrape_tournament_results)

    # --- scrape-conference-seeds ---
    seed_scrape_parser = subparsers.add_parser(
        "scrape-conference-seeds",
        help="Scrape conference tournament seeds from ESPN API",
    )
    seed_scrape_parser.add_argument(
        "--year", type=int, default=2026,
        help="Season year (default: 2026)",
    )
    seed_scrape_parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON file path (default: data/raw/seed_overrides_{year}.json)",
    )
    seed_scrape_parser.add_argument(
        "--conferences", nargs="+", default=None,
        help="Specific conferences to scrape (e.g., ACC SEC B10). Default: all",
    )
    seed_scrape_parser.add_argument(
        "--cache-dir", default="data/cache",
        help="Cache directory for scraped data (default: data/cache)",
    )
    seed_scrape_parser.set_defaults(func=run_scrape_conference_seeds)
