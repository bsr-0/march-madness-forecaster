"""Scraper exports."""

from .espn_picks import CBSPicksScraper, ESPNPicksScraper, YahooPicksScraper, aggregate_consensus
from .cbbpy_rosters import CBBpyRosterScraper
from .kenpom import KenPomScraper
from .ncaa_stats import NCAAStatsScraper
from .player_metrics import PlayerMetricsScraper
from .shotquality import ShotQualityScraper
from .shotquality_proxy import OpenShotQualityProxyBuilder
from .sports_reference import SportsReferenceScraper
from .tournament_bracket import TournamentSeedScraper
from .torvik import BartTorvikScraper
from .transfer_portal import TransferPortalScraper

__all__ = [
    "BartTorvikScraper",
    "CBBpyRosterScraper",
    "KenPomScraper",
    "ShotQualityScraper",
    "OpenShotQualityProxyBuilder",
    "NCAAStatsScraper",
    "PlayerMetricsScraper",
    "SportsReferenceScraper",
    "TournamentSeedScraper",
    "TransferPortalScraper",
    "ESPNPicksScraper",
    "YahooPicksScraper",
    "CBSPicksScraper",
    "aggregate_consensus",
]
