"""Tests for the cache of HBR per-subject parameters.

``HBR.per_subject_params`` keeps the last result and uses it again for the
same model, posterior, X and batch effects. These tests check that a changed
input never gets an old result: every cached result must equal the result
of a fresh computation (DETERMINISTIC class, rtol 1e-10).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pcntoolkit.math_functions import likelihood as likelihood_module
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model import hbr as hbr_module
from test.test_golden._common import (
    assert_deterministic,
    copy_model,
    load_npz,
    to_normdata,
)

GOLDEN: dict[str, np.ndarray] = load_npz("hbr")


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Count the PyMC model builds; start and end with an empty cache."""
    calls: list[int] = []
    real = likelihood_module.Likelihood.create_model_with_data

    def counting(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(
        likelihood_module.Likelihood, "create_model_with_data", counting
    )
    hbr_module.clear_param_cache()
    yield calls
    hbr_module.clear_param_cache()


def _model(name: str, tmp_path: Path) -> NormativeModel:
    model = copy_model(f"hbr_{name}", tmp_path / name)
    model.saveresults = model.saveplots = model.evaluate_model = False
    return model


def _inputs(model: NormativeModel, name: str, shift: float = 0.0):
    """Return scaled X, be, Y of the stored test data; ``shift`` moves X."""
    data = to_normdata(
        "test",
        GOLDEN[f"{name}_test_X"] + shift,
        GOLDEN[f"{name}_test_be"],
        GOLDEN[f"{name}_test_Y"],
    )
    model.preprocess(data)
    rv = model.response_vars[0]
    X, be, _, Y, _ = model.extract_data(data.sel(response_vars=rv))
    return model[rv], X, be, Y


def _z(n: int, value: float) -> xr.DataArray:
    return xr.DataArray(np.full(n, value), dims=("observations",))


def _fresh_backward(hbr, X, be, value: float) -> np.ndarray:
    hbr_module.clear_param_cache()
    return hbr.backward(X, be, _z(X.shape[0], value)).values


def test_039_hbrCache_should_buildOnce_when_sameDataIsUsedAgain(
    tmp_path: Path, builds: list[int]
) -> None:
    """
    Arrange: a saved HBR model and its test data.
    Act: backward at 3 Z values, then forward.
    Assert: one model build, and each result equals a fresh computation.
    """
    hbr, X, be, Y = _inputs(_model("Normal", tmp_path), "Normal")
    cached = [hbr.backward(X, be, _z(X.shape[0], v)).values for v in (-1, 0, 1)]
    z = hbr.forward(X, be, Y).values
    assert len(builds) == 1
    for v, values in zip((-1, 0, 1), cached, strict=True):
        assert_deterministic(values, _fresh_backward(hbr, X, be, v), near_zero=True)
    hbr_module.clear_param_cache()
    assert_deterministic(z, hbr.forward(X, be, Y).values, near_zero=True)


def test_040_hbrCache_should_rebuild_when_covariatesChange(
    tmp_path: Path, builds: list[int]
) -> None:
    """
    Arrange: one model; the test data, and the same data with X shifted.
    Act: backward on the first data, then on the shifted data.
    Assert: two builds; the second result equals a fresh computation.
    """
    model = _model("Normal", tmp_path)
    hbr, X, be, _ = _inputs(model, "Normal")
    _, X2, be2, _ = _inputs(model, "Normal", shift=5.0)
    first = hbr.backward(X, be, _z(X.shape[0], 1.0)).values
    second = hbr.backward(X2, be2, _z(X2.shape[0], 1.0)).values
    assert len(builds) == 2
    assert not np.array_equal(first, second)
    assert_deterministic(second, _fresh_backward(hbr, X2, be2, 1.0), near_zero=True)


def test_041_hbrCache_should_rebuild_when_batchEffectsChange(
    tmp_path: Path, builds: list[int]
) -> None:
    """
    Arrange: one model; the test data, and the same data with the levels
        of each batch effect reversed (all levels stay valid).
    Act: backward on both.
    Assert: two builds; the second result equals a fresh computation.
    """
    hbr, X, be, _ = _inputs(_model("Normal", tmp_path), "Normal")
    # Reverse the levels of each batch effect column; all stay valid.
    be2 = be.copy(data=be.values.max(axis=0) - be.values)
    assert not np.array_equal(be.values, be2.values)
    hbr.backward(X, be, _z(X.shape[0], 1.0))
    second = hbr.backward(X, be2, _z(X.shape[0], 1.0)).values
    assert len(builds) == 2
    assert_deterministic(second, _fresh_backward(hbr, X, be2, 1.0), near_zero=True)


def test_042_hbrCache_should_rebuild_when_modelOrPosteriorChanges(
    tmp_path: Path, builds: list[int]
) -> None:
    """
    Arrange: two models with different likelihoods, and a copy of the first
        model's posterior.
    Act: backward with model A, model B, model A again, then model A with a
        new idata object.
    Assert: a build for every call, and each result equals a fresh one.
    """
    a, X, be, _ = _inputs(_model("Normal", tmp_path), "Normal")
    b, Xb, beb, _ = _inputs(_model("SHASHb", tmp_path), "SHASHb")
    a.backward(X, be, _z(X.shape[0], 1.0))
    b_values = b.backward(Xb, beb, _z(Xb.shape[0], 1.0)).values
    a_values = a.backward(X, be, _z(X.shape[0], 1.0)).values
    a.idata = copy.deepcopy(a.idata)
    a.backward(X, be, _z(X.shape[0], 1.0))
    assert len(builds) == 4
    assert_deterministic(b_values, _fresh_backward(b, Xb, beb, 1.0), near_zero=True)
    assert_deterministic(a_values, _fresh_backward(a, X, be, 1.0), near_zero=True)


def test_043_hbrCache_should_beReadOnlyAndCleared_when_predicted(
    tmp_path: Path, builds: list[int]
) -> None:
    """
    Arrange: a saved HBR model and its test data.
    Act: per_subject_params, then NormativeModel.predict.
    Assert: the cached arrays are read-only, and predict frees the cache.
    """
    model = _model("Normal", tmp_path)
    hbr, X, be, Y = _inputs(model, "Normal")
    for array in hbr.per_subject_params(X, be, Y):
        with pytest.raises(ValueError):
            array.values[0, 0] = 0.0
    data = to_normdata(
        "test",
        GOLDEN["Normal_test_X"],
        GOLDEN["Normal_test_be"],
        GOLDEN["Normal_test_Y"],
    )
    model.predict(data)
    assert hbr_module._PARAM_CACHE == {}
