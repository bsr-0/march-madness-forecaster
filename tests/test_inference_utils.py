"""Tests for src.pipeline.stages.inference — extracted inference utilities."""

import numpy as np
import pytest

from src.pipeline.stages.inference import (
    apply_seed_prior,
    embedding_probability,
    symmetrize_probability,
)


# --- embedding_probability ---

def test_embedding_probability_both_none():
    assert embedding_probability(None, None) == 0.5

def test_embedding_probability_one_none():
    assert embedding_probability(np.array([1.0]), None) == 0.5

def test_embedding_probability_fallback():
    v1 = np.array([1.0, 0.5, 0.0])
    v2 = np.array([0.0, 0.5, 1.0])
    prob = embedding_probability(v1, v2, projection_model=None)
    assert 0.0 <= prob <= 1.0

def test_embedding_probability_symmetric_fallback():
    v1 = np.array([1.0, 0.5])
    v2 = np.array([0.5, 1.0])
    p_fwd = embedding_probability(v1, v2)
    p_rev = embedding_probability(v2, v1)
    # Fallback should be roughly symmetric
    assert abs(p_fwd + p_rev - 1.0) < 0.3


# --- symmetrize_probability ---

def test_symmetrize_exact():
    assert symmetrize_probability(0.7, 0.3) == 0.7

def test_symmetrize_asymmetric():
    # If forward=0.65 and reverse=0.40, symmetrized = (0.65 + 0.60) / 2 = 0.625
    result = symmetrize_probability(0.65, 0.40)
    assert abs(result - 0.625) < 1e-9


# --- apply_seed_prior ---

def test_seed_prior_lower_seed_favoured():
    # Seed 1 vs seed 16: prior should push toward team 1
    result = apply_seed_prior(0.5, seed1=1, seed2=16, weight=0.1)
    assert result > 0.5

def test_seed_prior_equal_seeds():
    result = apply_seed_prior(0.6, seed1=8, seed2=8, weight=0.1)
    # Prior is 0.5 when seeds equal, so result pulled slightly toward 0.5
    assert result < 0.6

def test_seed_prior_zero_weight():
    result = apply_seed_prior(0.7, seed1=1, seed2=16, weight=0.0)
    assert abs(result - 0.7) < 1e-9
