"""Tests for coverage audit resolver behavior."""

from src.data.coverage_audit import build_game_id_resolver


def test_resolver_handles_alias_exact_match():
    team_metrics = {
        "texas_a_m": {},
        "a_m_corpus_christi": {},
    }
    resolver = build_game_id_resolver(team_metrics, {"games": []})
    assert resolver("a_m") == "texas_a_m"


def test_resolver_handles_mascot_suffix():
    team_metrics = {
        "a_m_corpus_christi": {},
        "texas_a_m": {},
    }
    resolver = build_game_id_resolver(team_metrics, {"games": []})
    assert resolver("a_m_corpus_christi_islanders") == "a_m_corpus_christi"


def test_resolver_avoids_ambiguous_alias_prefix():
    team_metrics = {
        "miami__fl": {},
        "miami__oh": {},
    }
    resolver = build_game_id_resolver(team_metrics, {"games": []})
    assert resolver("miami_redhawks") is None


def test_resolver_prefers_exact_or_unique_prefix():
    team_metrics = {
        "alabama": {},
        "alabama_birmingham": {},
    }
    resolver = build_game_id_resolver(team_metrics, {"games": []})
    assert resolver("alabama") == "alabama"
    assert resolver("alabama_birmingham_blazers") == "alabama_birmingham"
