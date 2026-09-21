"""Shared protocol guards for staged VMOD experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path


MARGIN_SUMMARY_SCHEMA_VERSION = 2


def read_audited_margin_summary(path: str | Path) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    version = int(payload.get("summary_schema_version", 0))
    if version < MARGIN_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"Margin summary schema {version} is older than required "
            f"{MARGIN_SUMMARY_SCHEMA_VERSION}: {source}"
        )
    if payload.get("selected_margin_pu") is None:
        raise RuntimeError("No voltage margin passed the frozen calibration gate")
    return payload


def wait_for_audited_margin_summary(
    path: str | Path, poll_seconds: float = 60.0
) -> dict:
    source = Path(path)
    while True:
        if source.exists():
            try:
                return read_audited_margin_summary(source)
            except (json.JSONDecodeError, ValueError):
                pass
        time.sleep(poll_seconds)
