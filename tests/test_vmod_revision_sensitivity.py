from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_vmod_capacity_sensitivity import validate_table
from prepare_vmod_heterogeneous_profiles import (
    autocorrelated_deviations,
    weighted_center,
)


def test_declared_revision_paths_are_monotone() -> None:
    table = pd.read_csv(
        Path(__file__).parents[1] / "data" / "vmod" / "revision_capacity_paths.csv"
    )
    validate_table(table)


def test_heterogeneity_generator_is_reproducible_and_centered() -> None:
    first = autocorrelated_deviations(64, 4, 0.9, np.random.default_rng(7))
    second = autocorrelated_deviations(64, 4, 0.9, np.random.default_rng(7))
    np.testing.assert_allclose(first, second)
    weights = np.linspace(0.5, 1.5, 64)[:, None] * np.ones((1, 4))
    centered = weighted_center(first, weights)
    np.testing.assert_allclose((centered * weights).sum(axis=0), 0.0, atol=1e-10)
