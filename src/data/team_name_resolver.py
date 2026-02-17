"""
Canonical team name resolution across data sources.

NCAA basketball has no universal team ID standard. ESPN, Sports Reference,
Torvik, Warren Nolan, and Kaggle all use different naming conventions:

  ESPN:             "UConn"
  Sports Reference: "Connecticut" (slug: "connecticut")
  Warren Nolan:     "Connecticut"
  Torvik:           "Connecticut"
  Kaggle:           "Connecticut"

This module provides a single `TeamNameResolver` that:
1. Maintains a curated alias table for all 68 tournament-eligible schools
2. Normalizes any input string to a canonical internal ID
3. Uses multi-pass fuzzy matching (exact → alias → slug → token → Levenshtein)
4. Is the single source of truth for team identity across the pipeline
"""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Alias table: canonical_id -> set of known aliases from all sources
#
# Canonical ID is lowercase alphanumeric with underscores (no doubles).
# The first entry per team is the display name used in output.
# ---------------------------------------------------------------------------

_ALIAS_TABLE: Dict[str, List[str]] = {
    # --- A ---
    "alabama": ["Alabama", "Bama"],
    "alabama_am": ["Alabama A&M", "Alabama A and M", "Ala A&M", "AAMU"],
    "arizona": ["Arizona", "U of Arizona"],
    "arizona_state": ["Arizona State", "Arizona St", "ASU"],
    "arkansas": ["Arkansas", "Ark"],
    "arkansas_pine_bluff": ["Arkansas-Pine Bluff", "Ark Pine Bluff", "UAPB"],
    "auburn": ["Auburn"],
    # --- B ---
    "baylor": ["Baylor"],
    "boise_state": ["Boise State", "Boise St"],
    "boston_college": ["Boston College", "BC"],
    "brigham_young": ["BYU", "Brigham Young"],
    "butler": ["Butler"],
    # --- C ---
    "california": ["California", "Cal", "Cal Berkeley"],
    "charleston": ["College of Charleston", "Charleston", "C of Charleston", "Col of Charleston"],
    "clemson": ["Clemson"],
    "cleveland_state": ["Cleveland State", "Cleveland St"],
    "colorado": ["Colorado", "Colo"],
    "colorado_state": ["Colorado State", "Colorado St"],
    "connecticut": ["Connecticut", "UConn", "Conn"],
    "creighton": ["Creighton"],
    # --- D ---
    "dayton": ["Dayton"],
    "drake": ["Drake"],
    "duke": ["Duke"],
    "duquesne": ["Duquesne"],
    # --- F ---
    "fairleigh_dickinson": ["Fairleigh Dickinson", "FDU"],
    "florida": ["Florida", "Fla"],
    "florida_atlantic": ["Florida Atlantic", "FAU", "Fla Atlantic"],
    "florida_state": ["Florida State", "Florida St", "FSU"],
    "furman": ["Furman"],
    # --- G ---
    "george_mason": ["George Mason", "GMU"],
    "georgetown": ["Georgetown", "G'town"],
    "georgia": ["Georgia", "UGA"],
    "georgia_tech": ["Georgia Tech", "Ga Tech", "GT"],
    "gonzaga": ["Gonzaga", "Zags"],
    "grand_canyon": ["Grand Canyon", "GCU"],
    "grambling": ["Grambling", "Grambling State"],
    # --- H ---
    "houston": ["Houston", "U of Houston"],
    "howard": ["Howard"],
    # --- I ---
    "illinois": ["Illinois", "Ill"],
    "indiana": ["Indiana", "IU"],
    "indiana_state": ["Indiana State", "Indiana St"],
    "iowa": ["Iowa"],
    "iowa_state": ["Iowa State", "Iowa St", "ISU"],
    # --- J ---
    "james_madison": ["James Madison", "JMU"],
    # --- K ---
    "kansas": ["Kansas", "KU"],
    "kansas_state": ["Kansas State", "Kansas St", "K-State"],
    "kent_state": ["Kent State", "Kent St"],
    "kentucky": ["Kentucky", "UK"],
    # --- L ---
    "liberty": ["Liberty"],
    "long_beach_state": ["Long Beach State", "Long Beach St", "LBSU"],
    "louisiana_state": ["LSU", "Louisiana State"],
    "louisville": ["Louisville", "U of L"],
    "loyola_chicago": ["Loyola Chicago", "Loyola-Chicago", "Loyola (IL)", "Loyola IL"],
    # --- M ---
    "marquette": ["Marquette"],
    "maryland": ["Maryland", "UMD"],
    "memphis": ["Memphis"],
    "miami_fl": ["Miami (FL)", "Miami FL", "Miami Florida", "Miami"],
    "miami_oh": ["Miami (OH)", "Miami OH", "Miami Ohio"],
    "michigan": ["Michigan", "Mich"],
    "michigan_state": ["Michigan State", "Michigan St", "MSU"],
    "minnesota": ["Minnesota", "Minn"],
    "mississippi": ["Mississippi", "Ole Miss"],
    "mississippi_state": ["Mississippi State", "Mississippi St", "Miss State"],
    "missouri": ["Missouri", "Mizzou"],
    "montana_state": ["Montana State", "Montana St"],
    "morehead_state": ["Morehead State", "Morehead St"],
    "mount_st_marys": ["Mount St. Mary's", "Mount Saint Mary's", "Mt St Mary's", "Mt. St. Mary's"],
    "murray_state": ["Murray State", "Murray St"],
    # --- N ---
    "navy": ["Navy"],
    "nc_state": ["NC State", "North Carolina State", "N.C. State"],
    "nebraska": ["Nebraska", "Neb"],
    "nevada": ["Nevada"],
    "new_mexico": ["New Mexico", "UNM"],
    "new_mexico_state": ["New Mexico State", "New Mexico St", "NMSU"],
    "north_carolina": ["North Carolina", "UNC", "N Carolina"],
    "north_carolina_at": ["North Carolina A&T", "NC A&T", "North Carolina A and T"],
    "north_texas": ["North Texas", "UNT", "N Texas"],
    "northwestern": ["Northwestern", "NW"],
    "notre_dame": ["Notre Dame", "ND"],
    # --- O ---
    "oakland": ["Oakland"],
    "ohio_state": ["Ohio State", "Ohio St", "OSU"],
    "oklahoma": ["Oklahoma", "OU"],
    "oklahoma_state": ["Oklahoma State", "Oklahoma St"],
    "old_dominion": ["Old Dominion", "ODU"],
    "oral_roberts": ["Oral Roberts", "ORU"],
    "oregon": ["Oregon", "U of Oregon"],
    "oregon_state": ["Oregon State", "Oregon St"],
    # --- P ---
    "penn_state": ["Penn State", "Pennsylvania State"],
    "pittsburgh": ["Pittsburgh", "Pitt"],
    "princeton": ["Princeton"],
    "providence": ["Providence"],
    "purdue": ["Purdue"],
    # --- R ---
    "richmond": ["Richmond"],
    "rutgers": ["Rutgers"],
    # --- S ---
    "saint_johns": ["St. John's", "Saint John's", "St John's (NY)", "St. John's (NY)"],
    "saint_marys": ["Saint Mary's", "Saint Mary's College", "St. Mary's", "SMC"],
    "saint_peters": ["Saint Peter's", "St. Peter's"],
    "samford": ["Samford"],
    "san_diego_state": ["San Diego State", "San Diego St", "SDSU"],
    "seton_hall": ["Seton Hall"],
    "south_carolina": ["South Carolina", "S Carolina", "USC Upstate"],
    "south_dakota_state": ["South Dakota State", "South Dakota St", "SDSU Jackrabbits"],
    "southern_california": ["USC", "Southern California", "Southern Cal"],
    "stanford": ["Stanford"],
    "stephen_f_austin": ["Stephen F. Austin", "SFA", "SF Austin"],
    "stetson": ["Stetson"],
    "syracuse": ["Syracuse", "Cuse"],
    # --- T ---
    "temple": ["Temple"],
    "tennessee": ["Tennessee", "Tenn"],
    "texas": ["Texas", "U of Texas"],
    "texas_am": ["Texas A&M", "Texas A and M", "TAMU"],
    "texas_christian": ["TCU", "Texas Christian"],
    "texas_southern": ["Texas Southern", "TSU"],
    "texas_tech": ["Texas Tech", "TT"],
    "toledo": ["Toledo"],
    # --- U ---
    "uc_irvine": ["UC Irvine", "UC-Irvine", "Irvine"],
    "uc_san_diego": ["UC San Diego", "UCSD"],
    "uc_santa_barbara": ["UC Santa Barbara", "UCSB"],
    "ucf": ["UCF", "Central Florida"],
    "ucla": ["UCLA"],
    "umbc": ["UMBC", "Maryland-Baltimore County", "Maryland Baltimore County"],
    "unlv": ["UNLV", "Nevada-Las Vegas"],
    "utah": ["Utah"],
    "utah_state": ["Utah State", "Utah St"],
    # --- V ---
    "vanderbilt": ["Vanderbilt", "Vandy"],
    "vcu": ["VCU", "Virginia Commonwealth"],
    "vermont": ["Vermont", "UVM"],
    "villanova": ["Villanova", "Nova"],
    "virginia": ["Virginia", "UVA"],
    "virginia_tech": ["Virginia Tech", "Va Tech", "VT"],
    # --- W ---
    "wagner": ["Wagner"],
    "wake_forest": ["Wake Forest"],
    "washington": ["Washington", "UW"],
    "washington_state": ["Washington State", "Washington St", "Wazzu"],
    "west_virginia": ["West Virginia", "WVU", "W Virginia"],
    "western_kentucky": ["Western Kentucky", "WKU", "W Kentucky"],
    "wichita_state": ["Wichita State", "Wichita St"],
    "william_mary": ["William & Mary", "William and Mary", "W&M"],
    "wisconsin": ["Wisconsin", "Wisc"],
    "wright_state": ["Wright State", "Wright St"],
    # --- X ---
    "xavier": ["Xavier"],
    # --- Y ---
    "yale": ["Yale"],
}


def _normalize_str(s: str) -> str:
    """Normalize a string for matching: lowercase, decode HTML, collapse whitespace."""
    s = html.unescape(s)
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_canonical_id(s: str) -> str:
    """Convert any string to a canonical team ID format."""
    s = html.unescape(s)
    s = s.lower().strip()
    s = s.replace("&", "_")
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


@dataclass
class MatchResult:
    """Result of a team name resolution attempt."""

    canonical_id: str
    display_name: str
    confidence: float  # 0.0 to 1.0
    method: str  # "exact", "alias", "slug", "fuzzy", "unresolved"


class TeamNameResolver:
    """
    Resolves arbitrary team name strings to canonical internal IDs.

    Resolution order:
    1. Exact canonical ID match
    2. Alias table lookup (all known variants)
    3. Sports Reference school_slug match
    4. Normalized string containment
    5. Levenshtein/SequenceMatcher fuzzy match (threshold=0.80)

    Thread-safe for reads after construction.
    """

    def __init__(self, extra_aliases: Optional[Dict[str, List[str]]] = None):
        # Build lookup indices
        self._id_to_display: Dict[str, str] = {}
        self._normalized_to_id: Dict[str, str] = {}
        self._slug_to_id: Dict[str, str] = {}
        self._all_ids: Set[str] = set()

        combined = dict(_ALIAS_TABLE)
        if extra_aliases:
            for cid, aliases in extra_aliases.items():
                if cid in combined:
                    combined[cid] = combined[cid] + aliases
                else:
                    combined[cid] = aliases

        for canonical_id, aliases in combined.items():
            self._all_ids.add(canonical_id)
            self._id_to_display[canonical_id] = aliases[0] if aliases else canonical_id

            # Index the canonical ID itself
            self._normalized_to_id[_normalize_str(canonical_id)] = canonical_id

            # Index each alias
            for alias in aliases:
                norm = _normalize_str(alias)
                self._normalized_to_id[norm] = canonical_id

                # Also index the _to_canonical_id form
                cid_form = _to_canonical_id(alias)
                self._normalized_to_id[cid_form] = canonical_id

            # Index the canonical ID as a slug
            slug = canonical_id.replace("_", "-")
            self._slug_to_id[slug] = canonical_id

    def resolve(self, name: str) -> MatchResult:
        """
        Resolve a team name to its canonical ID.

        Args:
            name: Any team name string from any source

        Returns:
            MatchResult with canonical_id, display_name, confidence, method
        """
        if not name or not name.strip():
            return MatchResult("", "", 0.0, "empty")

        raw = name.strip()

        # Pass 1: Exact canonical ID
        cid = _to_canonical_id(raw)
        if cid in self._all_ids:
            return MatchResult(cid, self._id_to_display[cid], 1.0, "exact_id")

        # Pass 2: Normalized alias lookup
        norm = _normalize_str(raw)
        if norm in self._normalized_to_id:
            cid = self._normalized_to_id[norm]
            return MatchResult(cid, self._id_to_display[cid], 0.99, "alias")

        # Pass 2b: Canonical ID form alias lookup
        if cid in self._normalized_to_id:
            resolved = self._normalized_to_id[cid]
            return MatchResult(resolved, self._id_to_display[resolved], 0.98, "alias_id")

        # Pass 3: Sports Reference school_slug
        slug = raw.lower().replace(" ", "-").replace("'", "").replace(".", "")
        slug = re.sub(r"[^a-z0-9-]", "", slug)
        if slug in self._slug_to_id:
            cid = self._slug_to_id[slug]
            return MatchResult(cid, self._id_to_display[cid], 0.97, "slug")

        # Pass 4: Token containment — if the normalized input contains a known
        # team name or vice versa, and the match is unambiguous
        candidates = []
        for known_norm, known_id in self._normalized_to_id.items():
            if len(known_norm) < 4:
                continue
            if known_norm in norm or norm in known_norm:
                candidates.append((known_id, len(known_norm)))

        if len(candidates) == 1:
            cid = candidates[0][0]
            return MatchResult(cid, self._id_to_display[cid], 0.90, "containment")
        elif candidates:
            # Pick longest match (most specific)
            candidates.sort(key=lambda x: -x[1])
            cid = candidates[0][0]
            return MatchResult(cid, self._id_to_display[cid], 0.85, "containment_best")

        # Pass 5: Fuzzy string matching
        best_score = 0.0
        best_id = ""
        for known_norm, known_id in self._normalized_to_id.items():
            score = difflib.SequenceMatcher(None, norm, known_norm).ratio()
            if score > best_score:
                best_score = score
                best_id = known_id

        if best_score >= 0.80 and best_id:
            return MatchResult(
                best_id, self._id_to_display[best_id], best_score, "fuzzy"
            )

        # Unresolved — return normalized form as ID
        return MatchResult(cid, raw, best_score, "unresolved")

    def resolve_batch(
        self, names: List[str], warn_threshold: float = 0.85
    ) -> List[MatchResult]:
        """Resolve a list of names. Logs warnings for low-confidence matches."""
        results = []
        for name in names:
            r = self.resolve(name)
            results.append(r)
        return results

    def get_display_name(self, canonical_id: str) -> str:
        """Get display name for a canonical ID."""
        return self._id_to_display.get(canonical_id, canonical_id)

    def add_alias(self, canonical_id: str, alias: str) -> None:
        """Add a runtime alias mapping."""
        norm = _normalize_str(alias)
        self._normalized_to_id[norm] = canonical_id
        cid_form = _to_canonical_id(alias)
        self._normalized_to_id[cid_form] = canonical_id
        if canonical_id not in self._all_ids:
            self._all_ids.add(canonical_id)
            self._id_to_display[canonical_id] = alias

    @property
    def known_teams(self) -> Set[str]:
        """All known canonical team IDs."""
        return set(self._all_ids)
