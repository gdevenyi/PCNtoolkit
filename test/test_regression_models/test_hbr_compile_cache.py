"""Tests for the reuse of the compiled nutpie model in HBR fits.

HBR replaces ``nutpie.compile_pymc_model`` while ``pm.sample`` runs, so that
response variables with the same model graph share one numba compile. With
the same seed, the result must be bit-identical to a fresh compile.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import nutpie
import pymc as pm
import pytest
import xarray as xr

from pcntoolkit.math_functions.likelihood import (
    SHASHbLikelihood,
    get_default_normal_likelihood,
)
from pcntoolkit.math_functions.prior import make_prior
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model import hbr as hbr_module
from pcntoolkit.regression_model.hbr import HBR
from test.test_golden._common import to_normdata

N = 120
SAMPLING = {"draws": 20, "tune": 20, "chains": 2, "cores": 1, "progressbar": False}


@pytest.fixture
def compiles(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Count the real nutpie compiles; start and end with an empty cache."""
    calls: list[int] = []
    real = nutpie.compile_pymc_model

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(nutpie, "compile_pymc_model", counting)
    hbr_module.clear_compile_cache()
    yield calls
    hbr_module.clear_compile_cache()
    assert nutpie.compile_pymc_model is counting  # the replacement was undone


def _shashb() -> SHASHbLikelihood:
    return SHASHbLikelihood(
        make_prior("mu", dist_name="Normal", dist_params=(0.0, 1.0)),
        make_prior(
            "sigma", dist_name="Normal", dist_params=(1.0, 1.0), mapping="softplus"
        ),
        make_prior("epsilon", dist_name="Normal", dist_params=(0.0, 1.0)),
        make_prior(
            "delta",
            dist_name="Normal",
            dist_params=(1.0, 1.0),
            mapping="softplus",
            mapping_params=(0.0, 3.0, 0.6),
        ),
    )


def _shuffle_tail(rng: np.random.Generator, n: int, keep: int = 10) -> np.ndarray:
    return np.concatenate([np.arange(keep), keep + rng.permutation(n - keep)])


def _data(n: int = N, n_vars: int = 2, permute: bool = False):
    """Synthetic data. With ``permute``, the rows of X and of the batch effects
    are shuffled separately and Y is new: other values per subject, but the
    same ranges, levels and scaling (so the same basis knots and graph)."""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(20, 80, n), rng.uniform(0, 1, n)])
    be = np.column_stack(
        [rng.choice(["a", "b"], n), rng.choice(["s1", "s2", "s3"], n)]
    ).astype(str)
    if permute:
        # Keep the first rows, so the batch levels appear in the same order
        # (the same be_maps); shuffle the other rows.
        rng = np.random.default_rng(1)
        X, be = X[_shuffle_tail(rng, n)], be[_shuffle_tail(rng, n)]
    Y = 0.05 * X[:, :1] + rng.normal(0, 1, (n, n_vars))
    return to_normdata("train", X, be, Y)


def _fit_inputs(
    likelihood, data, save_dir: Path | None = None
) -> tuple[NormativeModel, list]:
    """Return the NormativeModel and (hbr, X, be, be_maps, Y) per variable."""
    model = NormativeModel(
        HBR("t", likelihood=likelihood, **SAMPLING),
        savemodel=False,
        evaluate_model=False,
        saveresults=False,
        saveplots=False,
        save_dir=str(save_dir) if save_dir else None,
    )
    model.register_data_info(data)
    model.preprocess(data)
    inputs = []
    for rv in model.response_vars:
        X, be, be_maps, Y, _ = model.extract_data(data.sel(response_vars=rv))
        inputs.append((model[rv], X, be, be_maps, Y))
    return model, inputs


def _sample(hbr: HBR, X, be, be_maps, Y, seed: int) -> xr.DataTree:
    """What HBR.fit does, with a seed."""
    hbr.be_maps = copy.deepcopy(be_maps)
    pymc_model = hbr.likelihood.compile(X, be, hbr.be_maps, Y)
    with pymc_model:
        return hbr._run_inference(random_seed=seed)


def _fresh(hbr: HBR, X, be, be_maps, Y, seed: int) -> xr.DataTree:
    """pm.sample on a new model, without the cache."""
    pymc_model = copy.deepcopy(hbr.likelihood).compile(X, be, be_maps, Y)
    with pymc_model:
        return pm.sample(
            random_seed=seed, nuts_sampler="nutpie", init="auto", **SAMPLING
        )


def _assert_same(a: xr.DataTree, b: xr.DataTree) -> None:
    for group in ("posterior", "sample_stats", "observed_data", "constant_data"):
        assert a[group].to_dataset().equals(b[group].to_dataset()), group


@pytest.mark.parametrize("likelihood", [get_default_normal_likelihood, _shashb])
def test_hbrCompileCache_should_beBitIdentical_when_reusedForAnotherVariable(
    compiles: list[int], likelihood
) -> None:
    """
    Arrange: one fit with 2 response variables.
    Act: sample variable 1, then variable 2 with the cached compile.
    Assert: 1 compile; variable 2 equals a fresh pm.sample with the same seed
        (posterior, sample_stats, observed_data and constant_data).
    """
    _, inputs = _fit_inputs(likelihood(), _data())
    _sample(*inputs[0], seed=1)
    second = _sample(*inputs[1], seed=7)
    assert len(compiles) == 1
    hbr_module.clear_compile_cache()
    _assert_same(second, _fresh(*inputs[1], seed=7))


def test_hbrCompileCache_should_swapAllData_when_otherDataHasSameShape(
    compiles: list[int],
) -> None:
    """
    Arrange: two fits with the same shapes, ranges and levels, but other X,
        batch effects and Y for each subject.
    Act: sample the first, then the second (cached compile).
    Assert: 1 compile, all data arrays differ, and the second equals a fresh
        pm.sample.
    """
    _, first = _fit_inputs(get_default_normal_likelihood(), _data())
    _, other = _fit_inputs(get_default_normal_likelihood(), _data(permute=True))
    for a, b in zip(first[0][1:], other[0][1:], strict=True):
        if isinstance(a, xr.DataArray):
            assert a.shape == b.shape and not np.array_equal(a.values, b.values)
    assert first[0][3] == other[0][3]  # the same be_maps
    _sample(*first[0], seed=1)
    second = _sample(*other[0], seed=7)
    assert len(compiles) == 1
    hbr_module.clear_compile_cache()
    _assert_same(second, _fresh(*other[0], seed=7))


def test_hbrCompileCache_should_compileAgain_when_graphOrShapeChanges(
    compiles: list[int],
) -> None:
    """
    Arrange: Normal data, the same with more observations, and SHASHb.
    Act: sample each one.
    Assert: a compile for each (3), because the keys differ.
    """
    _, normal = _fit_inputs(get_default_normal_likelihood(), _data())
    _, longer = _fit_inputs(get_default_normal_likelihood(), _data(n=N + 30))
    _, shash = _fit_inputs(_shashb(), _data())
    for inputs in (normal, longer, shash):
        _sample(*inputs[0], seed=1)
    assert len(compiles) == 3


def test_hbrCompileCache_should_compileEachVariable_when_transferred(
    compiles: list[int], tmp_path: Path
) -> None:
    """
    Arrange: a fitted 2-variable model.
    Act: transfer it to new data.
    Assert: each transferred variable is compiled (its priors come from its
        own posterior, so the graphs differ).
    """
    model, _ = _fit_inputs(get_default_normal_likelihood(), _data(), tmp_path / "a")
    model.fit(_data())
    assert len(compiles) == 1
    model.transfer(_data(), save_dir=str(tmp_path / "b"), freedom=1)
    assert len(compiles) == 3
    assert hbr_module._COMPILE_CACHE == {}


def test_hbrCompileCache_should_beClearedAndUseRealCompile_when_fitEnds(
    compiles: list[int], tmp_path: Path
) -> None:
    """
    Arrange: a 2-variable fit.
    Act: NormativeModel.fit, then compile another model directly.
    Assert: 1 compile in the fit, the cache is empty after it, and a call
        outside HBR sampling uses the real compile.
    """
    model, _ = _fit_inputs(get_default_normal_likelihood(), _data(), tmp_path)
    model.fit(_data())
    assert len(compiles) == 1
    assert hbr_module._COMPILE_CACHE == {}
    assert hbr_module._PATCH_STATE["depth"] == 0
    with pm.Model() as other:
        pm.Normal("x")
    nutpie.compile_pymc_model(other)
    assert len(compiles) == 2
