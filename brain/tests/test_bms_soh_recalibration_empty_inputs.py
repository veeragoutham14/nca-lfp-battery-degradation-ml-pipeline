from __future__ import annotations

import pandas as pd

from src.bms_soh_recalibration_analysis import (
    _empty_controls_like,
    _empty_events,
    detect_raw_discharge_evidence_batched,
)


def _detect(tmp_path, anchors: pd.DataFrame, anchor_column: str) -> pd.DataFrame:
    result, metrics = detect_raw_discharge_evidence_batched(
        tmp_path,
        serial="100000099",
        chemistry="NCA",
        anchors=anchors,
        anchor_column=anchor_column,
        lookback_hours=72,
        start_soc_pct=95.0,
        end_soc_pct=5.0,
        maximum_integrated_gap_seconds=60,
        rest_current_threshold_a=0.5,
        minimum_coverage_fraction=0.8,
        minimum_discharge_dominance_fraction=0.8,
    )
    assert metrics["anchor_batches"] == 0
    return result


def test_no_events_still_yield_the_raw_detection_schema(tmp_path) -> None:
    events = _detect(tmp_path, _empty_events(), "event_time_utc")
    assert events.empty
    assert {"raw_discharge_detection_status", "raw_end_time_utc"} <= set(events.columns)


def test_no_matched_controls_keep_an_anchor_column(tmp_path) -> None:
    events = _detect(tmp_path, _empty_events(), "event_time_utc")
    controls = _detect(tmp_path, _empty_controls_like(events), "anchor_time_utc")
    assert controls.empty
    assert "anchor_time_utc" in controls.columns
    assert "event_time_utc" not in controls.columns
    assert "raw_discharge_detection_status" in controls.columns
