"""Tests for the chunked HBR predict (DETERMINISTIC class).

``HBR.generic_MCMC_apply`` applies the likelihood function and the mean over
posterior samples to chunks of observations, and parameters that do not vary
over observations stay (1, sample).
- The chunking alone must not change any bit.
- Against the old path (every parameter repeated to (observations, sample),
  then one ``xr.apply_ufunc(fn, ...).mean("sample")``), the result must be in
  the DETERMINISTIC class. It is bit-identical where the output of ``fn`` has
  the same memory layout as before; otherwise numpy sums the samples in
  another order (a difference of about 1e-16).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pcntoolkit.math_functions.likelihood import (
    ZeroInflatedNegativeBinomialLikelihood as ZINBLikelihood,
)
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model import hbr as hbr_module
from test.test_golden._common import (
    assert_deterministic,
    copy_model,
    load_npz,
    to_normdata,
)

GOLDEN: dict[str, np.ndarray] = load_npz("hbr")
MODELS = ["Normal", "SHASHb", "beta", "ZINB"]


@pytest.fixture(autouse=True)
def empty_cache() -> Iterator[None]:
    hbr_module.clear_param_cache()
    yield
    hbr_module.clear_param_cache()


def _inputs(name: str, tmp_path: Path):
    """Return the HBR of response variable 0 and its scaled X, be and Y."""
    model: NormativeModel = copy_model(f"hbr_{name}", tmp_path / name)
    model.saveresults = model.saveplots = model.evaluate_model = False
    data = to_normdata(
        "test",
        GOLDEN[f"{name}_test_X"],
        GOLDEN[f"{name}_test_be"],
        GOLDEN[f"{name}_test_Y"],
    )
    model.preprocess(data)
    rv = model.response_vars[0]
    X, be, _, Y, _ = model.extract_data(data.sel(response_vars=rv))
    return model[rv], X, be, Y


def _old_apply(hbr, X, be, Y, fn, kwargs) -> np.ndarray:
    """The path before chunking: repeat every parameter to (N, S)."""
    n_obs = X.shape[0]
    arrays = [
        xr.DataArray(
            np.repeat(a.values, n_obs, axis=0) if a.shape[0] == 1 else a.values,
            dims=("observations", "sample"),
        )
        for a in hbr.per_subject_params(X, be, Y)
    ]
    return xr.apply_ufunc(fn, *arrays, kwargs=kwargs).mean(dim="sample").values


def _calls(hbr, Y) -> dict[str, tuple]:
    """fn and kwargs of forward, backward (Z = 1.5) and yhat."""
    z = np.full((Y.shape[0], 1), 1.5)
    if isinstance(hbr.likelihood, ZINBLikelihood):
        z = np.full((Y.shape[0], 1), -0.5)  # forward of ZINB uses an unseeded rng
    return {
        "backward": (hbr.likelihood.backward, {"Z": z}),
        "yhat": (hbr.likelihood.yhat, {}),
    } | (
        {}
        if isinstance(hbr.likelihood, ZINBLikelihood)
        else {"forward": (hbr.likelihood.forward, {"Y": Y.values[:, None]})}
    )


def test_044_chunkBounds_should_coverAllRowsWithoutOneRowChunks_when_split() -> None:
    """
    Arrange: several numbers of observations and chunk sizes.
    Act: _chunk_bounds.
    Assert: the chunks are in order, cover every row once, and none has
        exactly 1 row (unless there is only 1 observation).
    """
    for n_obs in (1, 2, 3, 7, 10, 11, 100, 101):
        for elements in (1, 6, 20, 2**22):
            old = hbr_module.PREDICT_CHUNK_ELEMENTS
            hbr_module.PREDICT_CHUNK_ELEMENTS = elements
            try:
                bounds = hbr_module._chunk_bounds(n_obs, 3)
            finally:
                hbr_module.PREDICT_CHUNK_ELEMENTS = old
            assert bounds[0][0] == 0 and bounds[-1][1] == n_obs
            assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
            sizes = [hi - lo for lo, hi in bounds]
            assert min(sizes) >= min(2, n_obs)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("chunk_rows", [2, 3, 7])
def test_045_chunkedApply_should_notChangeAnyBit_when_chunkSizeChanges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, chunk_rows: int
) -> None:
    """
    Arrange: a saved HBR model of each likelihood and its test data.
    Act: generic_MCMC_apply for forward, backward and yhat, with chunks of 2,
        3 or 7 rows (with a remainder) and with one chunk for all rows.
    Assert: the results are bit-identical.
    """
    hbr, X, be, Y = _inputs(name, tmp_path)
    n_samples = max(a.shape[1] for a in hbr.per_subject_params(X, be, Y))
    for call, (fn, kwargs) in _calls(hbr, Y).items():
        monkeypatch.setattr(hbr_module, "PREDICT_CHUNK_ELEMENTS", 2**40)
        whole = hbr.generic_MCMC_apply(X, be, Y, fn, kwargs).values
        elements = chunk_rows * n_samples
        monkeypatch.setattr(hbr_module, "PREDICT_CHUNK_ELEMENTS", elements)
        chunked = hbr.generic_MCMC_apply(X, be, Y, fn, kwargs).values
        np.testing.assert_array_equal(chunked, whole, call)


# (model, call) pairs whose fn output had another memory layout in the old
# path, so the mean over samples is summed in another order.
NOT_BIT_IDENTICAL = {("beta", "yhat"), ("ZINB", "yhat")}


@pytest.mark.parametrize("name", MODELS)
def test_048_chunkedApply_should_matchOldPath_when_defaultChunks(
    tmp_path: Path, name: str
) -> None:
    """
    Arrange: a saved HBR model of each likelihood and its test data.
    Act: generic_MCMC_apply, and the old repeat-and-apply_ufunc path, for
        forward, backward and yhat.
    Assert: DETERMINISTIC class; bit-identical except the listed pairs.
    """
    hbr, X, be, Y = _inputs(name, tmp_path)
    for call, (fn, kwargs) in _calls(hbr, Y).items():
        new = hbr.generic_MCMC_apply(X, be, Y, fn, kwargs).values
        old = _old_apply(hbr, X, be, Y, fn, kwargs)
        assert_deterministic(new, old, name=f"{name} {call}")
        if (name, call) not in NOT_BIT_IDENTICAL:
            np.testing.assert_array_equal(new, old, f"{name} {call}")


def test_046_perSubjectParams_should_keepSampleOnlyParamsSmall_when_notPerObservation(
    tmp_path: Path,
) -> None:
    """
    Arrange: the saved SHASHb model (fixed epsilon and delta).
    Act: per_subject_params.
    Assert: mu (linear in X) is (N, S); sigma, epsilon and delta (one value
        per sample in this model) are (1, S).
    """
    hbr, X, be, Y = _inputs("SHASHb", tmp_path)
    shapes = [a.shape for a in hbr.per_subject_params(X, be, Y)]
    n_obs, n_samples = X.shape[0], shapes[0][1]
    assert shapes == [(n_obs, n_samples)] + [(1, n_samples)] * 3


def test_047_zinbBackward_should_matchRepeatedParams_when_paramsAreOneRow() -> None:
    """
    Arrange: ZINB parameters as (1, S) arrays and as the same values repeated
        to (N, S); Z values that use both the zero mass and the NB part.
    Act: ZINBLikelihood.backward.
    Assert: the results are equal.
    """
    rng = np.random.default_rng(0)
    n_obs, n_samples = 50, 30
    mu = rng.uniform(0.5, 5, (1, n_samples))
    alpha = rng.uniform(0.5, 2, (1, n_samples))
    psi = rng.uniform(0.3, 0.9, (1, n_samples))
    z = rng.normal(0, 1.5, (n_obs, 1))
    lik = ZINBLikelihood.__new__(ZINBLikelihood)
    small = lik.backward(mu, alpha, psi, Z=z)
    full = lik.backward(*(np.repeat(a, n_obs, axis=0) for a in (mu, alpha, psi)), Z=z)
    assert small.shape == (n_obs, n_samples)
    np.testing.assert_array_equal(small, full)


def test_049_predict_should_keepOutputOrder_when_cacheIsFreedBeforeLogp(
    tmp_path: Path,
) -> None:
    """
    Arrange: a saved HBR model and its test data.
    Act: NormativeModel.predict (Yhat is computed before logp, then moved).
    Assert: the outputs are in the old order, and the cache is empty.
    """
    model: NormativeModel = copy_model("hbr_Normal", tmp_path / "Normal")
    model.saveresults = model.saveplots = model.evaluate_model = False
    data = to_normdata(
        "test",
        GOLDEN["Normal_test_X"],
        GOLDEN["Normal_test_be"],
        GOLDEN["Normal_test_Y"],
    )
    model.predict(data)
    outputs = ["Z", "centiles", "baseline_logp", "logp", "Yhat"]
    assert [v for v in data.data_vars if v in outputs] == outputs
    assert hbr_module._PARAM_CACHE == {}
