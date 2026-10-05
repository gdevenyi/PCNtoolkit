"""Golden-value tests for ``NormativeModel.predict`` as one call.

``predict`` runs compute_zscores, compute_centiles, compute_baseline_logp,
compute_logp and compute_yhat, then the Evaluator, then writes the result
files. All outputs come from a saved model, so the DETERMINISTIC class
applies. Plots are switched off (slow, and not a numeric output).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from test.test_golden._common import (
    PREDICT_CASES,
    assert_deterministic,
    copy_model,
    full_predict_outputs,
    load_npz,
    to_normdata,
)

GOLDEN: dict[str, np.ndarray] = load_npz("predict")
SOURCES: dict[str, dict[str, np.ndarray]] = {
    npz: load_npz(npz) for npz, _, _ in PREDICT_CASES.values()
}
ARRAY_KEYS: tuple[str, ...] = ("Z", "centiles", "baseline_logp", "logp", "yhat")


def _test_data(name: str) -> NormData:
    """Return a fresh NormData of the held-out data of one saved model."""
    npz, prefix, _ = PREDICT_CASES[name]
    src = SOURCES[npz]
    return to_normdata(
        "test", src[f"{prefix}test_X"], src[f"{prefix}test_be"], src[f"{prefix}test_Y"]
    )


@pytest.fixture(scope="module", params=list(PREDICT_CASES))
def predicted(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, dict[str, Any]]:
    """Run ``predict`` once with one saved model copied to a temporary dir."""
    name: str = request.param
    model = copy_model(name, tmp_path_factory.mktemp(name))
    return name, full_predict_outputs(model, _test_data(name))


@pytest.mark.parametrize("key", ARRAY_KEYS)
def test_030_predict_should_matchGolden_when_calledOnce(
    predicted: tuple[str, dict[str, Any]], key: str
) -> None:
    """
    Arrange: a saved golden model (BLR plain, HBR Normal), copied to a
        temporary directory, with evaluation and result saving switched on.
    Act: NormativeModel.predict on the held-out data.
    Assert: Z, centiles, baseline_logp, logp and yhat equal the golden values
        (signed, so atol is on).
    """
    name, actual = predicted
    assert_deterministic(
        actual[key], GOLDEN[f"{name}_{key}"], near_zero=True, name=f"{name}/{key}"
    )


@pytest.mark.parametrize("key", ("Z", "centiles", "logp", "yhat"))
def test_031_predict_should_equalSeparateCalls_when_calledOnce(
    predicted: tuple[str, dict[str, Any]], key: str
) -> None:
    """
    Arrange: as test 030.
    Act: NormativeModel.predict.
    Assert: each output equals the golden value of the separate compute_*
        call (``blr.npz``/``hbr.npz``), so both paths give the same numbers.
    """
    name, actual = predicted
    npz, _, pred_prefix = PREDICT_CASES[name]
    assert_deterministic(
        actual[key],
        SOURCES[npz][f"{pred_prefix}{key}"],
        near_zero=True,
        name=f"{name}/{key}",
    )


def test_032_predictStatistics_should_matchGoldenByName_when_evaluated(
    predicted: tuple[str, dict[str, Any]],
) -> None:
    """
    Arrange: as test 030.
    Act: NormativeModel.predict with evaluate_model on.
    Assert: the same set of statistic names as golden, and each statistic
        equal to its golden value (Rho, skew and kurtosis are signed, so atol
        is on). Statistics are matched by name, not position.
    """
    name, actual = predicted
    expected_names = [str(s) for s in GOLDEN[f"{name}_statistic_names"]]
    actual_names = [str(s) for s in actual["statistic_names"]]
    assert set(actual_names) == set(expected_names), name
    for stat in expected_names:
        assert_deterministic(
            actual["statistics"][..., actual_names.index(stat)],
            GOLDEN[f"{name}_statistics"][..., expected_names.index(stat)],
            near_zero=True,
            name=f"{name}/{stat}",
        )


def test_033_predict_should_writeResultFiles_when_saveresultsIsOn(
    predicted: tuple[str, dict[str, Any]],
) -> None:
    """
    Arrange: as test 030.
    Act: NormativeModel.predict with saveresults on.
    Assert: the same result files as golden (Z, centiles, logp, statistics).
        The file contents are not compared.
    """
    name, actual = predicted
    assert actual["results_files"] == [str(f) for f in GOLDEN[f"{name}_results_files"]]


@pytest.mark.parametrize("name", list(PREDICT_CASES))
def test_034_computeBaselineLogp_should_matchGolden_when_calledAlone(
    tmp_path: Path, name: str
) -> None:
    """
    Arrange: a saved golden model (BLR plain, HBR Normal) and its held-out
        data.
    Act: compute_baseline_logp alone (Gaussian with the mean and SD of the
        scaled Y).
    Assert: equal to the golden baseline_logp of predict (signed, so atol is
        on).
    """
    model = copy_model(name, tmp_path)
    data = model.compute_baseline_logp(_test_data(name))
    assert_deterministic(
        data["baseline_logp"].values,
        GOLDEN[f"{name}_baseline_logp"],
        near_zero=True,
        name=name,
    )
