"""Golden-value tests for NormData scaling, Evaluator metrics and batch-effect helpers.

All are deterministic, so they use the DETERMINISTIC tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from test.test_golden._common import (
    assert_deterministic,
    batch_combination_outputs,
    evaluator_outputs,
    extract_and_reshape_outputs,
    load_npz,
    normdata_scaling_outputs,
)

GOLDEN: dict[str, np.ndarray] = load_npz("normdata_eval")

SCALING_KEYS: list[str] = [
    f"nd_{d}_{v}" for d in ("fwd", "bwd") for v in ("X", "Y", "Yhat", "centiles")
]


@pytest.fixture(scope="module")
def scaling_actual() -> dict[str, np.ndarray]:
    """Recompute the NormData scaling outputs once per module."""
    return normdata_scaling_outputs(GOLDEN)


@pytest.fixture(scope="module")
def evaluator_actual() -> dict[str, np.ndarray]:
    """Recompute the Evaluator outputs once per module."""
    return evaluator_outputs(GOLDEN)


@pytest.mark.parametrize("key", SCALING_KEYS)
def test_008_normDataScaling_should_matchGolden_when_scaledForwardAndBack(
    scaling_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: a NormData with X, Y, Yhat and centiles, and fitted scalers.
    Act: scale_forward, then scale_backward.
    Assert: each stage equals the golden values (standardised values cross 0).
    """
    assert_deterministic(scaling_actual[key], GOLDEN[key], near_zero=True, name=key)


def test_009_evaluator_should_matchGolden_when_givenFixedPredictions(
    evaluator_actual: dict[str, np.ndarray],
) -> None:
    """
    Arrange: a NormData with fixed Y, Yhat, Z, centiles, logp and
        baseline_logp for two response variables and two batch-effect dims.
    Act: Evaluator().evaluate (all statistics).
    Assert: the same statistics, each equal to its golden value.
    """
    np.testing.assert_array_equal(evaluator_actual["eval_names"], GOLDEN["eval_names"])
    # Rho_p is ~1e-47 here; signed metrics (MSLL, Skewness, Kurtosis) can be ~0.
    assert_deterministic(
        evaluator_actual["eval_statistics"],
        GOLDEN["eval_statistics"],
        near_zero=True,
        name="statistics",
    )


def test_010_evaluatorBic_should_matchGolden_when_givenFixedPredictions(
    evaluator_actual: dict[str, np.ndarray],
) -> None:
    """
    Arrange: the same prediction dataset as test 009.
    Act: Evaluator()._evaluate_bic per response variable (BIC is not part of
        evaluate()).
    Assert: equal to the golden values.
    """
    assert_deterministic(evaluator_actual["eval_bic"], GOLDEN["eval_bic"], name="BIC")


def test_011_iterBatchCombinations_should_matchGolden_when_combosMissing() -> None:
    """
    Arrange: batch effects where one site/sex pair never occurs, and a level
        list that includes a site with no subjects.
    Act: list all combinations and masks.
    Assert: the same combinations in the same order, with identical masks.
    """
    actual = batch_combination_outputs(GOLDEN)
    np.testing.assert_array_equal(actual["ibc_combos"], GOLDEN["ibc_combos"])
    np.testing.assert_array_equal(actual["ibc_masks"], GOLDEN["ibc_masks"])


@pytest.mark.parametrize("key", ["ear_global", "ear_per_subject"])
def test_012_extractAndReshape_should_matchGolden_when_givenBothInputShapes(
    key: str,
) -> None:
    """
    Arrange: a posterior-predictive dataset with one global (sample,) variable
        and one per-subject (observations, sample) variable.
    Act: HBR.extract_and_reshape on each.
    Assert: equal to the golden arrays (a pure reshape, so exact).
    """
    actual = extract_and_reshape_outputs(GOLDEN)
    np.testing.assert_array_equal(actual[key], GOLDEN[key])
