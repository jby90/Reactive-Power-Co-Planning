import json

import pytest

import vmod_protocol


def test_audited_margin_summary_requires_current_schema(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({"summary_schema_version": 1, "selected_margin_pu": 0.003}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="older than required"):
        vmod_protocol.read_audited_margin_summary(path)


def test_audited_margin_summary_requires_selected_margin(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "summary_schema_version": vmod_protocol.MARGIN_SUMMARY_SCHEMA_VERSION,
                "selected_margin_pu": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No voltage margin"):
        vmod_protocol.read_audited_margin_summary(path)


def test_audited_margin_summary_accepts_frozen_selection(tmp_path):
    payload = {
        "summary_schema_version": vmod_protocol.MARGIN_SUMMARY_SCHEMA_VERSION,
        "selected_margin_pu": 0.001,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert vmod_protocol.read_audited_margin_summary(path) == payload
