"""Canonical id -> Kaggle TeamID cascade, on a spellings table that mirrors Kaggle's."""

import pytest

from src.prediction.kaggle_bridge import build_bridge, canonical_to_kaggle_id

SPELLINGS = {
    "duke": 1181,
    "michigan state": 1277,
    "saint mary's ca": 1388,
    "texas a&m": 1401,
    "miami fl": 1274,
    "st john's": 1385,
    "umbc": 1420,
    "saint peter's": 1389,
    "loyola (il)": 1260,
    "north carolina a&t": 1299,
    "nevada-las-vegas": 1424,
    "stephen f. austin": 1372,
    "texas a&m-corpus christi": 1394,
    "mount st. mary's": 1291,
    "saint francis (pa)": 1384,
    "st francis (ny)": 1383,
}


@pytest.mark.parametrize(
    "canonical, expected",
    [
        ("duke", 1181),
        ("michigan_state", 1277),
        ("saint_mary_s__ca", 1388),
        ("texas_a_m", 1401),
        ("miami__fl", 1274),
        ("st__john_s__ny", 1385),
        ("maryland_baltimore_county", 1420),
        ("saint_peter_s", 1389),  # trailing possessive
        ("loyola__il", 1260),  # parenthesised suffix
        ("north_carolina_a_t", 1299),  # a&t
        ("nevada_las_vegas", 1424),  # hyphenated
        ("stephen_f_austin", 1372),  # alias
        ("texas_a_m_corpus_christi", 1394),  # alias
        ("mount_st__mary_s", 1291),  # alias
        ("saint_francis", 1384),  # alias picks the (PA) program
        ("no_such_team", None),
    ],
)
def test_cascade(canonical, expected):
    assert canonical_to_kaggle_id(canonical, SPELLINGS) == expected


def test_build_bridge_omits_unresolved_and_is_invertible():
    c2k, k2c = build_bridge(["duke", "no_such_team", "loyola__il"], SPELLINGS)
    assert c2k == {"duke": 1181, "loyola__il": 1260}
    assert k2c == {1181: "duke", 1260: "loyola__il"}
