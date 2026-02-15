"""Scraper exports."""

from .espn_picks import CBSPicksScraper, ESPNPicksScraper, YahooPicksScraper, aggregate_consensus
from .kenpom import KenPomScraper
from .ncaa_stats import NCAAStatsScraper
from .shotquality import ShotQualityScraper
from .sports_reference import SportsReferenceScraper
from .tournament_bracket import TournamentSeedScraper
from .torvik import BartTorvikScraper
from .transfer_portal import TransferPortalScraper

__all__ = [
    "BartTorvikScraper",
    "KenPomScraper",
    "ShotQualityScraper",
    "NCAAStatsScraper",
    "SportsReferenceScraper",
    "TournamentSeedScraper",
    "TransferPortalScraper",
    "ESPNPicksScraper",
    "YahooPicksScraper",
    "CBSPicksScraper",
    "aggregate_consensus",
]
