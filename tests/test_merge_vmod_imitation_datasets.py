from pathlib import Path

import numpy as np
import pytest

from merge_vmod_imitation_datasets import ARRAY_KEYS, merge_datasets


def write_dataset(path: Path, day: int) -> None:
    np.savez_compressed(
        path,
        observations=np.full((2, 4), day, dtype=np.float32),
        thetas=np.zeros((2, 3), dtype=np.float32),
        target_actions=np.zeros((2, 2), dtype=np.float32),
        day_indices=np.full(2, day, dtype=np.int32),
        candidate_ids=np.zeros(2, dtype=np.int16),
    )


def test_merge_preserves_canonical_arrays(tmp_path):
    base = tmp_path / "base.npz"
    augmentation = tmp_path / "augmentation.npz"
    write_dataset(base, 1)
    write_dataset(augmentation, 10001)

    merged = merge_datasets(base, augmentation)

    assert set(merged) == set(ARRAY_KEYS)
    assert merged["observations"].shape == (4, 4)
    assert set(merged["day_indices"].tolist()) == {1, 10001}


def test_merge_rejects_day_identifier_overlap(tmp_path):
    base = tmp_path / "base.npz"
    augmentation = tmp_path / "augmentation.npz"
    write_dataset(base, 1)
    write_dataset(augmentation, 1)

    with pytest.raises(ValueError, match="day identifiers overlap"):
        merge_datasets(base, augmentation)
