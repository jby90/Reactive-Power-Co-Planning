import numpy as np

import benchmark_vmod_scalability as scaling


def test_distribution_reports_requested_quantiles():
    result = scaling.distribution([1.0, 2.0, 3.0, 4.0])

    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert np.isclose(result["p95"], 3.85)
    assert result["maximum"] == 4.0


def test_synthetic_profiles_match_each_network():
    for env_id, expected_pv in ((33, 3), (69, 4), (118, 8)):
        load, pv = scaling.synthetic_profiles(env_id, 20)

        assert load.shape[0] == 21
        assert pv.shape == (21, expected_pv)
        assert load.ndim == 2
        assert np.all(load > 0)
        assert np.all(pv > 0)
