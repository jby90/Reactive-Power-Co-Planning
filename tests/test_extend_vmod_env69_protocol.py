from extend_vmod_env69_protocol import EXTENSION, build_extended_path
from vmod_current_study import ROOT


def test_env69_extension_is_nested_and_reaches_actor_domain_endpoint() -> None:
    source = ROOT / "runs" / "VMOD_PROTOCOL_MAIN_ENV69_20260920"
    path = build_extended_path(source)

    assert len(path) == 9 + len(EXTENSION)
    assert path.pv_s_scale.is_monotonic_increasing
    assert path.svc_q_scale.is_monotonic_increasing
    assert tuple(path.iloc[-1][["pv_s_scale", "svc_q_scale"]]) == (1.5, 1.5)
    assert path.path_index.tolist() == list(range(len(path)))
