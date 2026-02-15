"""Tests for EV bracket generation quality."""

from src.optimization.leverage import TeamMetadata, analyze_pool


def _build_probs():
    model_probs = {}
    public_probs = {}
    metadata = {}

    for region in ("East", "West", "South", "Midwest"):
        for seed in range(1, 17):
            team_id = f"{region.lower()}_{seed}"
            seed_strength = (17 - seed) / 16.0
            model_probs[team_id] = {
                "R64": min(0.99, 0.45 + 0.50 * seed_strength),
                "R32": min(0.95, 0.20 + 0.60 * seed_strength),
                "S16": min(0.85, 0.10 + 0.55 * seed_strength),
                "E8": min(0.70, 0.05 + 0.45 * seed_strength),
                "F4": min(0.50, 0.03 + 0.35 * seed_strength),
                "CHAMP": min(0.30, 0.01 + 0.22 * seed_strength),
            }
            public_probs[team_id] = {
                "R64": min(0.99, 0.50 + 0.45 * seed_strength),
                "R32": min(0.97, 0.25 + 0.55 * seed_strength),
                "S16": min(0.90, 0.12 + 0.48 * seed_strength),
                "E8": min(0.75, 0.06 + 0.36 * seed_strength),
                "F4": min(0.55, 0.03 + 0.28 * seed_strength),
                "CHAMP": min(0.32, 0.01 + 0.18 * seed_strength),
            }
            metadata[team_id] = TeamMetadata(
                team_name=team_id.upper(),
                seed=seed,
                region=region,
            )

    return model_probs, public_probs, metadata


def test_pareto_optimizer_generates_full_63_game_bracket():
    model_probs, public_probs, metadata = _build_probs()
    analysis = analyze_pool(
        pool_size=600,
        model_probs=model_probs,
        public_picks=public_probs,
        team_metadata=metadata,
    )
    assert analysis.pareto_brackets

    bracket = analysis.pareto_brackets[-1]
    assert bracket.champion
    assert len(bracket.final_four) == 4
    assert "CHAMP" in bracket.picks
    assert len(bracket.picks) >= 63
    assert any(k.startswith("R64_") for k in bracket.picks)
