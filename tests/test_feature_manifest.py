from src.governance.feature_manifest import build_feature_manifest


def test_feature_manifest_tracks_selected_and_dropped_features():
    manifest = build_feature_manifest(
        target_year=2026,
        model_complexity="standard",
        selection_method="fixed_domain_knowledge",
        original_features=["a", "b", "c"],
        selected_features=["a", "c"],
        training_years=[2016, 2017],
        dev_years=[2016, 2017],
        holdout_years=[2025],
    )

    payload = manifest.to_dict()

    assert payload["original_dim"] == 3
    assert payload["final_dim"] == 2
    assert payload["dropped_features"] == ["b"]
    assert payload["manifest_hash"]
