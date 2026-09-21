"""Finite-sample and independent-seed statistics for VMOD evidence."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta, t


def clopper_pearson_upper(events: int, trials: int, confidence: float = 0.95) -> float:
    """Return the exact one-sided binomial upper confidence limit."""
    if trials <= 0 or not 0 <= events <= trials:
        raise ValueError("Require 0 <= events <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if events == trials:
        return 1.0
    return float(beta.ppf(confidence, events + 1, trials - events))


def zero_event_sample_size(target_probability: float, confidence: float = 0.95) -> int:
    """Return the smallest n whose 0/n upper limit is below the target."""
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must lie strictly between zero and one")
    return int(
        np.floor(np.log(1.0 - confidence) / np.log(1.0 - target_probability)) + 1
    )


def mean_t_interval(
    values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return the mean and two-sided t interval over independent training seeds."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("At least one finite value is required")
    if values.size < 2:
        return float(values.mean()), float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(values.size))
    half = float(t.ppf(0.5 + confidence / 2.0, values.size - 1) * sem)
    return mean, mean - half, mean + half


def percentile_bootstrap_mean_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    *,
    resamples: int = 10_000,
    seed: int = 20260921,
) -> tuple[float, float, float]:
    """Return a reproducible percentile-bootstrap interval for a sample mean.

    This is used for bounded event-rate summaries whose independent units are
    training seeds or matched physical-shift realisations. Resampling whole
    units preserves their internal day/actor dependence and keeps the interval
    within the range of the observed unit-level rates.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("At least one finite value is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return mean, float(low), float(high)
