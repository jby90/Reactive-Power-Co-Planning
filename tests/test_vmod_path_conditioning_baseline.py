import pandas as pd
import pytest

from run_vmod_path_conditioning_baseline import (
    parse_candidate_ids,
    select_capacity_rows,
)


def test_parse_candidate_ids_preserves_order_and_removes_duplicates():
    assert parse_candidate_ids("4,2,3,2") == (4, 2, 3)
    with pytest.raises(ValueError, match="At least one"):
        parse_candidate_ids(" , ")


def test_select_capacity_rows_validates_requested_points():
    frame = pd.DataFrame(
        [
            {
                "path_index": index,
                "capacity_label": f"P{index:02d}",
                "pv_s_scale": 0.3,
                "svc_q_scale": 0.1 * index,
                "cap_total_mvar": 0.0,
            }
            for index in range(3)
        ]
    )
    rows = select_capacity_rows(frame, (2, 0))
    assert [row["path_index"] for row in rows] == [2, 0]
    with pytest.raises(ValueError, match="Unknown candidate"):
        select_capacity_rows(frame, (4,))
