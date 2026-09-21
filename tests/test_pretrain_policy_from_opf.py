import numpy as np
import pytest

from pretrain_policy_from_opf import resolve_frozen_validation_days


def test_frozen_validation_split_is_strict_by_default() -> None:
    with pytest.raises(ValueError, match="absent from the dataset"):
        resolve_frozen_validation_days(
            np.array([1, 2, 3]),
            np.array([1, 3, 4, 5]),
            allow_missing=False,
        )


def test_missing_validation_days_are_reported_not_reassigned() -> None:
    realised, missing = resolve_frozen_validation_days(
        np.array([1, 2, 3]),
        np.array([1, 3, 4, 5]),
        allow_missing=True,
    )

    assert realised.tolist() == [1, 3]
    assert missing == [2]
    assert 2 not in realised


def test_available_subset_must_still_contain_validation_and_training_days() -> None:
    with pytest.raises(ValueError, match="available validation and training"):
        resolve_frozen_validation_days(
            np.array([1, 2]),
            np.array([3, 4]),
            allow_missing=True,
        )
