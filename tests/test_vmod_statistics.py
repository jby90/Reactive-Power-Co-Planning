import numpy as np
import pytest

from vmod_statistics import percentile_bootstrap_mean_interval


def test_percentile_bootstrap_mean_interval_is_bounded_and_reproducible():
    values = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0 / 85.0])

    first = percentile_bootstrap_mean_interval(values)
    second = percentile_bootstrap_mean_interval(values)

    assert first == second
    assert first[0] == pytest.approx(values.mean())
    assert 0.0 <= first[1] <= first[2] <= 1.0


def test_percentile_bootstrap_mean_interval_validates_inputs():
    with pytest.raises(ValueError):
        percentile_bootstrap_mean_interval(np.asarray([]))
    with pytest.raises(ValueError):
        percentile_bootstrap_mean_interval(np.asarray([0.0, 1.0]), confidence=1.0)
    with pytest.raises(ValueError):
        percentile_bootstrap_mean_interval(np.asarray([0.0, 1.0]), resamples=0)
