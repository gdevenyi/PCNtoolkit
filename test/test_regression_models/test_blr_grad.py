"""Tests for the analytic gradient of the BLR objective (optimizer="l-bfgs-b-grad")."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from test.fixtures.blr_model_fixtures import *  # noqa: F403
from test.fixtures.norm_data_fixtures import *  # noqa: F403
from test.fixtures.path_fixtures import *  # noqa: F403

N, D, DV = 200, 6, 3

CONFIGS: dict[str, dict] = {
    "plain": {},
    "ard": {"ard": True},
    "hetero": {"heteroskedastic": True},
    "warp": {"heteroskedastic": True, "warp_name": "WarpSinhArcsinh"},
    "warp_reparam": {"warp_name": "WarpSinhArcsinh", "warp_reparam": True},
    "warp_compose": {"warp_name": "WarpCompose[WarpBoxCox,WarpAffine]"},
}


def _problem(config: str) -> tuple[BLR, np.ndarray, tuple]:
    """A BLR, a hyperparameter vector and (Phi, y, Phi_var) arrays."""
    rng = np.random.default_rng(0)
    blr = BLR("grad", **CONFIGS[config])
    blr.D, blr.var_D = D, DV
    Phi = np.column_stack([np.ones(N), rng.normal(size=(N, D - 1))])
    Phi_var = np.column_stack([np.ones(N), rng.normal(size=(N, DV - 1))])
    y = Phi.dot(rng.normal(size=D)) + rng.normal(0, 0.5, N)
    if "BoxCox" in CONFIGS[config].get("warp_name", ""):
        y = np.exp(y / 4)  # Box-Cox needs positive responses
    hyp = rng.normal(0, 0.3, blr.init_hyp().size)
    return blr, hyp, (Phi, y, Phi_var)


def _central_difference(
    f: Callable[[np.ndarray], float], hyp: np.ndarray
) -> np.ndarray:
    step = 1e-6
    grad = np.zeros(hyp.shape)
    for k in range(hyp.size):
        e = np.zeros(hyp.shape)
        e[k] = step
        grad[k] = (f(hyp + e) - f(hyp - e)) / (2 * step)
    return grad


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("norm", ["L1", "L2"])
def test_penalizedGrad_should_matchFiniteDifferences_when_givenAnyModelType(
    config: str, norm: str
) -> None:
    """
    Arrange: a BLR of each type, random data and random hyperparameters.
    Act: penalized_loglik_and_grad, and central differences of
        penalized_loglik.
    Assert: the values are equal, and the gradients agree to 1e-6.
    """
    blr, hyp, args = _problem(config)
    value, grad = blr.penalized_loglik_and_grad(hyp, *args, 0.1, norm)
    assert value == blr.penalized_loglik(hyp, *args, 0.1, norm)
    reference = _central_difference(
        lambda h: blr.penalized_loglik(h, *args, 0.1, norm), hyp
    )
    np.testing.assert_allclose(grad, reference, rtol=1e-6, atol=1e-6)


def test_loglikAndGrad_should_leavePosteriorAtHyp_when_warped() -> None:
    """
    Arrange: a warped BLR (the warp gradient evaluates nearby points).
    Act: loglik_and_grad.
    Assert: the stored posterior belongs to hyp, not to a nearby point.
    """
    blr, hyp, args = _problem("warp")
    blr.loglik_and_grad(hyp, *args)
    np.testing.assert_array_equal(blr.hyp, hyp)
    m = blr.m.copy()
    blr.hyp = np.full(hyp.shape, np.nan)
    blr.loglik(hyp, *args)
    np.testing.assert_array_equal(blr.m, m)


@pytest.mark.parametrize(
    "overrides",
    [
        {"heteroskedastic": False, "warp_name": None},
        {},  # heteroskedastic and warped (BLR_BASE_CONFIG)
    ],
)
def test_fit_should_reachFineStepOptimum_when_optimizerIsLbfgsbGrad(
    blr_model_factory: Callable,
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
    overrides: dict,
) -> None:
    """
    Arrange: the test data, and two BLR models.
    Act: fit with "l-bfgs-b-grad", and with "l-bfgs-b" and a small
        finite-difference step (1e-7), which reaches the same optimum.
    Assert: the objectives agree to 1e-6 (relative), and the analytic fit
        is not worse than the default fit (step 0.1).
    """
    rv = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(
        norm_data_from_arrays.sel(response_vars=rv)
    )
    fits = {}
    for name, extra in {
        "grad": {"optimizer": "l-bfgs-b-grad"},
        "fine": {"l_bfgs_b_epsilon": 1e-7},
        "default": {},
    }.items():
        blr = blr_model_factory(**overrides, **extra)
        blr.fit(X, be, be_maps, Y)
        assert blr.is_fitted
        fits[name] = blr.nlZ
    np.testing.assert_allclose(fits["grad"], fits["fine"], rtol=1e-6)
    assert fits["grad"] <= fits["default"] + 1e-9 * abs(fits["default"])
