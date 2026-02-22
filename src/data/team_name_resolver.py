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
1. Maintains a curated alias table covering all ~360 D1 programs
2. Normalizes any input string to a canonical internal ID
3. Uses multi-pass fuzzy matching (exact -> alias -> slug -> token -> Levenshtein)
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
#
# Coverage: all ~360 D1 programs plus common abbreviations, mascot-
# suffixed forms, and cross-source naming variants (ESPN, Sports
# Reference, Torvik, Warren Nolan, Kaggle, CBBpy).
# ---------------------------------------------------------------------------

_ALIAS_TABLE: Dict[str, List[str]] = {
    # --- A ---
    "abilene_christian": ["Abilene Christian", "ACU"],
    "air_force": ["Air Force"],
    "akron": ["Akron"],
    "alabama": ["Alabama", "Bama"],
    "alabama_a_m": ["Alabama A&M", "Alabama A and M", "Ala A&M", "AAMU"],
    "alabama_state": ["Alabama State", "Bama State"],
    "albany__ny": ["Albany (NY)", "Albany", "UAlbany"],
    "alcorn_state": ["Alcorn State", "Alcorn"],
    "american": ["American", "American University"],
    "appalachian_state": ["Appalachian State", "App State", "App St"],
    "arizona": ["Arizona", "U of Arizona"],
    "arizona_state": ["Arizona State", "Arizona St", "ASU"],
    "arkansas": ["Arkansas", "Ark"],
    "arkansas_pine_bluff": ["Arkansas-Pine Bluff", "Ark Pine Bluff", "UAPB"],
    "arkansas_state": ["Arkansas State", "Arkansas St"],
    "army": ["Army", "Army West Point"],
    "auburn": ["Auburn"],
    "austin_peay": ["Austin Peay", "Austin Peay State"],
    # --- B ---
    "ball_state": ["Ball State", "Ball St"],
    "baylor": ["Baylor"],
    "bellarmine": ["Bellarmine"],
    "belmont": ["Belmont"],
    "bethune_cookman": ["Bethune-Cookman", "Bethune Cookman", "B-CU"],
    "binghamton": ["Binghamton"],
    "birmingham_southern": ["Birmingham-Southern", "Birmingham Southern"],
    "boise_state": ["Boise State", "Boise St"],
    "boston_college": ["Boston College", "BC"],
    "boston_university": ["Boston University", "BU"],
    "bowling_green": ["Bowling Green", "BGSU"],
    "bradley": ["Bradley"],
    "brigham_young": ["BYU", "Brigham Young"],
    "brown": ["Brown"],
    "bryant": ["Bryant"],
    "bucknell": ["Bucknell"],
    "buffalo": ["Buffalo"],
    "butler": ["Butler"],
    # --- C ---
    "cal_poly": ["Cal Poly", "Cal Poly SLO"],
    "cal_state_bakersfield": ["Cal State Bakersfield", "CSU Bakersfield", "CSUB"],
    "cal_state_fullerton": ["Cal State Fullerton", "CSU Fullerton", "CSUF"],
    "cal_state_northridge": ["Cal State Northridge", "CSU Northridge", "CSUN"],
    "california": ["California", "Cal", "Cal Berkeley"],
    "california_baptist": ["California Baptist", "Cal Baptist", "CBU"],
    "campbell": ["Campbell"],
    "canisius": ["Canisius"],
    "centenary__la": ["Centenary (LA)", "Centenary"],
    "central_arkansas": ["Central Arkansas", "UCA"],
    "central_connecticut_state": ["Central Connecticut State", "Central Connecticut", "CCSU"],
    "central_michigan": ["Central Michigan", "CMU"],
    "charleston_southern": ["Charleston Southern", "CSU Charleston"],
    "charlotte": ["Charlotte", "UNC Charlotte"],
    "chattanooga": ["Chattanooga", "UTC"],
    "chicago_state": ["Chicago State"],
    "cincinnati": ["Cincinnati", "Cincy"],
    "clemson": ["Clemson"],
    "cleveland_state": ["Cleveland State", "Cleveland St"],
    "coastal_carolina": ["Coastal Carolina", "CCU"],
    "colgate": ["Colgate"],
    "college_of_charleston": ["College of Charleston", "Charleston", "C of Charleston", "Col of Charleston"],
    "colorado": ["Colorado", "Colo"],
    "colorado_state": ["Colorado State", "Colorado St"],
    "columbia": ["Columbia"],
    "connecticut": ["Connecticut", "UConn", "Conn"],
    "coppin_state": ["Coppin State", "Coppin"],
    "cornell": ["Cornell"],
    "creighton": ["Creighton"],
    # --- D ---
    "dartmouth": ["Dartmouth"],
    "davidson": ["Davidson"],
    "dayton": ["Dayton"],
    "delaware": ["Delaware"],
    "delaware_state": ["Delaware State", "Delaware St"],
    "denver": ["Denver"],
    "depaul": ["DePaul", "De Paul"],
    "detroit_mercy": ["Detroit Mercy", "Detroit"],
    "drake": ["Drake"],
    "drexel": ["Drexel"],
    "duke": ["Duke"],
    "duquesne": ["Duquesne"],
    # --- E ---
    "east_carolina": ["East Carolina", "ECU"],
    "east_tennessee_state": ["East Tennessee State", "East Tennessee St", "ETSU"],
    "east_texas_a_m": ["East Texas A&M", "Texas A&M-Commerce", "A&M-Commerce"],
    "eastern_illinois": ["Eastern Illinois", "EIU"],
    "eastern_kentucky": ["Eastern Kentucky", "EKU"],
    "eastern_michigan": ["Eastern Michigan", "EMU"],
    "eastern_washington": ["Eastern Washington", "EWU"],
    "elon": ["Elon"],
    "evansville": ["Evansville"],
    # --- F ---
    "fairfield": ["Fairfield"],
    "fairleigh_dickinson": ["Fairleigh Dickinson", "FDU"],
    "florida": ["Florida", "Fla"],
    "florida_a_m": ["Florida A&M", "Florida A and M", "FAMU"],
    "florida_atlantic": ["Florida Atlantic", "FAU", "Fla Atlantic"],
    "florida_gulf_coast": ["Florida Gulf Coast", "FGCU"],
    "florida_international": ["Florida International", "FIU"],
    "florida_state": ["Florida State", "Florida St", "FSU"],
    "fordham": ["Fordham"],
    "fresno_state": ["Fresno State", "Fresno St"],
    "furman": ["Furman"],
    # --- G ---
    "gardner_webb": ["Gardner-Webb", "Gardner Webb"],
    "george_mason": ["George Mason", "GMU"],
    "george_washington": ["George Washington", "GW", "GWU"],
    "georgetown": ["Georgetown", "G'town"],
    "georgia": ["Georgia", "UGA"],
    "georgia_southern": ["Georgia Southern", "Ga Southern"],
    "georgia_state": ["Georgia State", "Ga State"],
    "georgia_tech": ["Georgia Tech", "Ga Tech", "GT"],
    "gonzaga": ["Gonzaga", "Zags"],
    "grambling": ["Grambling", "Grambling State"],
    "grand_canyon": ["Grand Canyon", "GCU"],
    "green_bay": ["Green Bay", "Wisconsin-Green Bay", "UWGB"],
    # --- H ---
    "hampton": ["Hampton"],
    "hartford": ["Hartford"],
    "harvard": ["Harvard"],
    "hawaii": ["Hawaii", "Hawai'i"],
    "high_point": ["High Point"],
    "hofstra": ["Hofstra"],
    "holy_cross": ["Holy Cross"],
    "houston": ["Houston", "U of Houston"],
    "houston_christian": ["Houston Christian", "Houston Baptist", "HCU", "HBU"],
    "howard": ["Howard"],
    # --- I ---
    "idaho": ["Idaho"],
    "idaho_state": ["Idaho State", "Idaho St"],
    "illinois": ["Illinois", "Ill"],
    "illinois_chicago": ["Illinois-Chicago", "UIC"],
    "illinois_state": ["Illinois State", "Illinois St"],
    "incarnate_word": ["Incarnate Word", "UIW"],
    "indiana": ["Indiana", "IU"],
    "indiana_state": ["Indiana State", "Indiana St"],
    "iona": ["Iona"],
    "iowa": ["Iowa"],
    "iowa_state": ["Iowa State", "Iowa St", "ISU"],
    "iu_indy": ["IU Indy", "IU Indianapolis", "IUPUI"],
    # --- J ---
    "jackson_state": ["Jackson State", "Jackson St"],
    "jacksonville": ["Jacksonville", "JU"],
    "jacksonville_state": ["Jacksonville State", "Jacksonville St", "Jax State"],
    "james_madison": ["James Madison", "JMU"],
    # --- K ---
    "kansas": ["Kansas", "KU"],
    "kansas_city": ["Kansas City", "Missouri-Kansas City", "UMKC"],
    "kansas_state": ["Kansas State", "Kansas St", "K-State"],
    "kennesaw_state": ["Kennesaw State", "Kennesaw St"],
    "kent_state": ["Kent State", "Kent St"],
    "kentucky": ["Kentucky", "UK"],
    # --- L ---
    "la_salle": ["La Salle"],
    "lafayette": ["Lafayette"],
    "lamar": ["Lamar"],
    "le_moyne": ["Le Moyne"],
    "lehigh": ["Lehigh"],
    "liberty": ["Liberty"],
    "lindenwood": ["Lindenwood"],
    "lipscomb": ["Lipscomb"],
    "little_rock": ["Little Rock", "Arkansas-Little Rock", "UALR"],
    "long_beach_state": ["Long Beach State", "Long Beach St", "LBSU"],
    "long_island_university": ["Long Island University", "LIU", "Long Island"],
    "longwood": ["Longwood"],
    "louisiana": ["Louisiana", "Louisiana-Lafayette", "UL Lafayette"],
    "louisiana_monroe": ["Louisiana-Monroe", "UL Monroe", "ULM"],
    "louisiana_state": ["LSU", "Louisiana State"],
    "louisiana_tech": ["Louisiana Tech", "La Tech"],
    "louisville": ["Louisville", "U of L"],
    "loyola__il": ["Loyola (IL)", "Loyola Chicago", "Loyola-Chicago"],
    "loyola__md": ["Loyola (MD)", "Loyola Maryland"],
    "loyola_marymount": ["Loyola Marymount", "LMU"],
    # --- M ---
    "maine": ["Maine"],
    "manhattan": ["Manhattan"],
    "marist": ["Marist"],
    "marquette": ["Marquette"],
    "marshall": ["Marshall"],
    "maryland": ["Maryland", "UMD"],
    "maryland_baltimore_county": ["Maryland-Baltimore County", "UMBC"],
    "maryland_eastern_shore": ["Maryland-Eastern Shore", "Maryland Eastern Shore", "UMES"],
    "massachusetts": ["Massachusetts", "UMass"],
    "massachusetts_lowell": ["Massachusetts-Lowell", "UMass Lowell"],
    "mcneese_state": ["McNeese State", "McNeese"],
    "memphis": ["Memphis"],
    "mercer": ["Mercer"],
    "mercyhurst": ["Mercyhurst"],
    "merrimack": ["Merrimack"],
    "miami__fl": ["Miami (FL)", "Miami FL", "Miami Florida", "Miami"],
    "miami__oh": ["Miami (OH)", "Miami OH", "Miami Ohio", "Miami of Ohio"],
    "michigan": ["Michigan", "Mich"],
    "michigan_state": ["Michigan State", "Michigan St", "MSU"],
    "middle_tennessee": ["Middle Tennessee", "MTSU", "Middle Tennessee State"],
    "milwaukee": ["Milwaukee", "Wisconsin-Milwaukee", "UW-Milwaukee"],
    "minnesota": ["Minnesota", "Minn"],
    "mississippi": ["Mississippi", "Ole Miss"],
    "mississippi_state": ["Mississippi State", "Mississippi St", "Miss State"],
    "mississippi_valley_state": ["Mississippi Valley State", "MVSU"],
    "missouri": ["Missouri", "Mizzou"],
    "missouri_state": ["Missouri State", "Missouri St"],
    "monmouth": ["Monmouth"],
    "montana": ["Montana"],
    "montana_state": ["Montana State", "Montana St"],
    "morehead_state": ["Morehead State", "Morehead St"],
    "morgan_state": ["Morgan State"],
    "mount_st__mary_s": ["Mount St. Mary's", "Mt. St. Mary's", "Mount Saint Mary's"],
    "murray_state": ["Murray State", "Murray St"],
    # --- N ---
    "navy": ["Navy"],
    "nc_state": ["NC State", "North Carolina State", "N.C. State"],
    "nebraska": ["Nebraska", "Neb"],
    "nevada": ["Nevada"],
    "nevada_las_vegas": ["Nevada-Las Vegas", "UNLV"],
    "new_hampshire": ["New Hampshire", "UNH"],
    "new_mexico": ["New Mexico", "UNM"],
    "new_mexico_state": ["New Mexico State", "New Mexico St", "NMSU"],
    "new_orleans": ["New Orleans", "UNO New Orleans"],
    "niagara": ["Niagara"],
    "nicholls_state": ["Nicholls State", "Nicholls"],
    "njit": ["NJIT", "New Jersey Tech"],
    "norfolk_state": ["Norfolk State"],
    "north_alabama": ["North Alabama", "UNA"],
    "north_carolina": ["North Carolina", "UNC", "N Carolina"],
    "north_carolina_a_t": ["North Carolina A&T", "NC A&T", "North Carolina A and T"],
    "north_carolina_central": ["North Carolina Central", "NC Central", "NCCU"],
    "north_dakota": ["North Dakota", "UND"],
    "north_dakota_state": ["North Dakota State", "NDSU"],
    "north_florida": ["North Florida", "UNF"],
    "north_texas": ["North Texas", "UNT", "N Texas"],
    "northeastern": ["Northeastern", "NEU"],
    "northern_arizona": ["Northern Arizona", "NAU"],
    "northern_colorado": ["Northern Colorado"],
    "northern_illinois": ["Northern Illinois", "NIU"],
    "northern_iowa": ["Northern Iowa", "UNI"],
    "northern_kentucky": ["Northern Kentucky", "NKU"],
    "northwestern": ["Northwestern", "NW"],
    "northwestern_state": ["Northwestern State"],
    "notre_dame": ["Notre Dame", "ND"],
    # --- O ---
    "oakland": ["Oakland"],
    "ohio": ["Ohio", "Ohio Bobcats"],
    "ohio_state": ["Ohio State", "Ohio St", "OSU"],
    "oklahoma": ["Oklahoma", "OU"],
    "oklahoma_state": ["Oklahoma State", "Oklahoma St"],
    "old_dominion": ["Old Dominion", "ODU"],
    "omaha": ["Omaha", "Nebraska-Omaha", "UNO"],
    "oral_roberts": ["Oral Roberts", "ORU"],
    "oregon": ["Oregon", "U of Oregon"],
    "oregon_state": ["Oregon State", "Oregon St"],
    # --- P ---
    "pacific": ["Pacific"],
    "penn_state": ["Penn State", "Pennsylvania State"],
    "pennsylvania": ["Pennsylvania", "Penn"],
    "pepperdine": ["Pepperdine"],
    "pittsburgh": ["Pittsburgh", "Pitt"],
    "portland": ["Portland"],
    "portland_state": ["Portland State"],
    "prairie_view": ["Prairie View", "Prairie View A&M", "PVAMU"],
    "presbyterian": ["Presbyterian", "Presbyterian College"],
    "princeton": ["Princeton"],
    "providence": ["Providence"],
    "purdue": ["Purdue"],
    "purdue_fort_wayne": ["Purdue Fort Wayne", "IPFW", "Fort Wayne"],
    # --- Q ---
    "queens__nc": ["Queens (NC)", "Queens University"],
    "quinnipiac": ["Quinnipiac"],
    # --- R ---
    "radford": ["Radford"],
    "rhode_island": ["Rhode Island", "URI"],
    "rice": ["Rice"],
    "richmond": ["Richmond"],
    "rider": ["Rider"],
    "robert_morris": ["Robert Morris", "RMU"],
    "rutgers": ["Rutgers"],
    # --- S ---
    "sacramento_state": ["Sacramento State", "Cal State Sacramento", "Sac State"],
    "sacred_heart": ["Sacred Heart", "SHU"],
    "saint_francis__pa": ["Saint Francis (PA)", "St. Francis (PA)"],
    "saint_joseph_s": ["Saint Joseph's", "St. Joseph's", "St. Joe's"],
    "saint_louis": ["Saint Louis", "St. Louis", "SLU"],
    "saint_mary_s__ca": ["Saint Mary's (CA)", "Saint Mary's", "St. Mary's (CA)", "St. Mary's", "SMC"],
    "saint_peter_s": ["Saint Peter's", "St. Peter's"],
    "sam_houston": ["Sam Houston", "Sam Houston State", "SHSU"],
    "samford": ["Samford"],
    "san_diego": ["San Diego"],
    "san_diego_state": ["San Diego State", "San Diego St", "SDSU"],
    "san_francisco": ["San Francisco", "USF"],
    "san_jose_state": ["San Jose State", "SJSU", "San Jose St"],
    "santa_clara": ["Santa Clara"],
    "savannah_state": ["Savannah State"],
    "seattle": ["Seattle", "Seattle U", "Seattle University"],
    "seton_hall": ["Seton Hall", "SHU"],
    "siena": ["Siena"],
    "siu_edwardsville": ["SIU Edwardsville", "Southern Illinois-Edwardsville", "SIUE"],
    "south_alabama": ["South Alabama", "USA"],
    "south_carolina": ["South Carolina", "S Carolina"],
    "south_carolina_state": ["South Carolina State", "SC State"],
    "south_carolina_upstate": ["South Carolina Upstate", "USC Upstate"],
    "south_dakota": ["South Dakota"],
    "south_dakota_state": ["South Dakota State", "South Dakota St", "SDSU Jackrabbits"],
    "south_florida": ["South Florida", "USF Bulls"],
    "southeast_missouri_state": ["Southeast Missouri State", "SEMO", "SE Missouri St"],
    "southeastern_louisiana": ["Southeastern Louisiana", "SE Louisiana"],
    "southern": ["Southern", "Southern University"],
    "southern_california": ["USC", "Southern California", "Southern Cal"],
    "southern_illinois": ["Southern Illinois", "SIU"],
    "southern_indiana": ["Southern Indiana", "USI"],
    "southern_methodist": ["Southern Methodist", "SMU"],
    "southern_mississippi": ["Southern Mississippi", "Southern Miss", "USM"],
    "southern_utah": ["Southern Utah", "SUU"],
    "st__bonaventure": ["St. Bonaventure", "Saint Bonaventure"],
    "st__francis__ny": ["St. Francis (NY)", "Saint Francis (NY)", "St. Francis Brooklyn"],
    "st__john_s__ny": ["St. John's (NY)", "St. John's", "Saint John's"],
    "st__thomas": ["St. Thomas", "Saint Thomas (MN)", "St. Thomas (MN)"],
    "stanford": ["Stanford"],
    "stephen_f__austin": ["Stephen F. Austin", "SFA", "SF Austin"],
    "stetson": ["Stetson"],
    "stonehill": ["Stonehill"],
    "stony_brook": ["Stony Brook"],
    "syracuse": ["Syracuse", "Cuse"],
    # --- T ---
    "tarleton_state": ["Tarleton State", "Tarleton"],
    "tcu": ["TCU", "Texas Christian"],
    "temple": ["Temple"],
    "tennessee": ["Tennessee", "Tenn"],
    "tennessee_martin": ["Tennessee-Martin", "UT Martin", "UTM"],
    "tennessee_state": ["Tennessee State", "Tennessee St"],
    "tennessee_tech": ["Tennessee Tech", "TTU"],
    "texas": ["Texas", "U of Texas"],
    "texas_a_m": ["Texas A&M", "Texas A and M", "TAMU"],
    "texas_a_m_corpus_christi": ["Texas A&M-Corpus Christi", "A&M-Corpus Christi", "TAMUCC"],
    "texas_rio_grande_valley": ["Texas-Rio Grande Valley", "UTRGV", "UT Rio Grande Valley"],
    "texas_southern": ["Texas Southern", "TSU"],
    "texas_state": ["Texas State", "Texas St"],
    "texas_tech": ["Texas Tech", "TT"],
    "the_citadel": ["The Citadel", "Citadel"],
    "toledo": ["Toledo"],
    "towson": ["Towson"],
    "troy": ["Troy"],
    "tulane": ["Tulane"],
    "tulsa": ["Tulsa"],
    # --- U ---
    "uab": ["UAB", "Alabama-Birmingham", "Alabama at Birmingham"],
    "uc_davis": ["UC Davis", "California-Davis"],
    "uc_irvine": ["UC Irvine", "UC-Irvine", "Irvine"],
    "uc_riverside": ["UC Riverside", "California-Riverside", "UCR"],
    "uc_san_diego": ["UC San Diego", "UCSD"],
    "uc_santa_barbara": ["UC Santa Barbara", "UCSB"],
    "ucf": ["UCF", "Central Florida"],
    "ucla": ["UCLA"],
    "unc_asheville": ["UNC Asheville", "North Carolina-Asheville"],
    "unc_greensboro": ["UNC Greensboro", "North Carolina-Greensboro", "UNCG"],
    "unc_wilmington": ["UNC Wilmington", "North Carolina-Wilmington", "UNCW"],
    "ut_arlington": ["UT Arlington", "Texas-Arlington", "UTA"],
    "utah": ["Utah"],
    "utah_state": ["Utah State", "Utah St"],
    "utah_tech": ["Utah Tech", "Dixie State"],
    "utah_valley": ["Utah Valley", "UVU"],
    "utep": ["UTEP", "Texas-El Paso", "UT El Paso"],
    "utsa": ["UTSA", "Texas-San Antonio", "UT San Antonio"],
    # --- V ---
    "valparaiso": ["Valparaiso", "Valpo"],
    "vanderbilt": ["Vanderbilt", "Vandy"],
    "vermont": ["Vermont", "UVM"],
    "villanova": ["Villanova", "Nova"],
    "virginia": ["Virginia", "UVA"],
    "virginia_commonwealth": ["Virginia Commonwealth", "VCU"],
    "virginia_tech": ["Virginia Tech", "Va Tech", "VT"],
    "vmi": ["VMI", "Virginia Military Institute", "Virginia Military"],
    # --- W ---
    "wagner": ["Wagner"],
    "wake_forest": ["Wake Forest"],
    "washington": ["Washington", "UW"],
    "washington_state": ["Washington State", "Washington St", "Wazzu"],
    "weber_state": ["Weber State", "Weber St"],
    "west_georgia": ["West Georgia", "UWG"],
    "west_virginia": ["West Virginia", "WVU", "W Virginia"],
    "western_carolina": ["Western Carolina", "WCU"],
    "western_illinois": ["Western Illinois", "WIU"],
    "western_kentucky": ["Western Kentucky", "WKU", "W Kentucky"],
    "western_michigan": ["Western Michigan", "WMU"],
    "wichita_state": ["Wichita State", "Wichita St"],
    "william___mary": ["William & Mary", "William and Mary", "W&M"],
    "winston_salem": ["Winston-Salem", "Winston-Salem State", "WSSU"],
    "winthrop": ["Winthrop"],
    "wisconsin": ["Wisconsin", "Wisc"],
    "wofford": ["Wofford"],
    "wright_state": ["Wright State", "Wright St"],
    "wyoming": ["Wyoming"],
    # --- X ---
    "xavier": ["Xavier"],
    # --- Y ---
    "yale": ["Yale"],
    "youngstown_state": ["Youngstown State", "YSU"],
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

        # Pass 3b: Prefix matching for CBBpy mascot-suffixed IDs
        # (e.g. "duke_blue_devils" → try "duke blue devils" → match "duke")
        # Progressively strip trailing tokens to find a match.
        norm_tokens = norm.split()
        if len(norm_tokens) >= 2:
            for trim in range(1, min(len(norm_tokens), 3)):
                prefix = " ".join(norm_tokens[:-trim])
                if len(prefix) >= 4 and prefix in self._normalized_to_id:
                    cid = self._normalized_to_id[prefix]
                    return MatchResult(cid, self._id_to_display[cid], 0.92, "prefix_strip")

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
