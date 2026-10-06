"""Tests for the posterior mean in BLR.post (one Cholesky solve)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from pcntoolkit.regression_model.blr import BLR

N, D, DV = 300, 6, 3

CONFIGS: dict[str, dict] = {
    "plain": {},
    "ard": {"ard": True},
    "hetero": {"heteroskedastic": True},
}


def _problem(config: str) -> tuple[BLR, np.ndarray, tuple]:
    """A BLR, a hyperparameter vector and (Phi, y, Phi_var) arrays."""
    rng = np.random.default_rng(0)
    blr = BLR("post", **CONFIGS[config])
    blr.D, blr.var_D = D, DV
    Phi = np.column_stack([np.ones(N), rng.normal(size=(N, D - 1))])
    Phi_var = np.column_stack([np.ones(N), rng.normal(size=(N, DV - 1))])
    y = Phi.dot(rng.normal(size=D)) + rng.normal(0, 0.5, N)
    hyp = rng.normal(0, 0.5, blr.init_hyp().size)
    return blr, hyp, (Phi, y, Phi_var)


def _reference_m(blr: BLR, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The posterior mean as computed before: solve with N right-hand sides."""
    invAXt = linalg.solve(blr.A, X.T)
    return (invAXt * blr.lambda_n_vec).dot(y)


@pytest.mark.parametrize("config", list(CONFIGS))
def test_post_should_matchSolveWithAllRightHandSides_when_givenAnyNoiseModel(
    config: str,
) -> None:
    """
    Arrange: a BLR with constant noise, ARD, or heteroskedastic noise.
    Act: post().
    Assert: m equals A^-1 X^T Lambda_n y from a general solve (rtol 1e-10).
    """
    blr, hyp, (X, y, var_X) = _problem(config)
    blr.post(hyp, X, y, var_X)
    np.testing.assert_allclose(blr.m, _reference_m(blr, X, y), rtol=1e-10)


def test_post_should_fallBackToLuSolve_when_choleskyFails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Arrange: a heteroskedastic BLR, and a Cholesky factorization that fails
        (as for an A that is not positive definite in floating point).
    Act: post().
    Assert: post does not raise, and m is the general-solve result for the
        new hyperparameters.
    """
    blr, hyp, (X, y, var_X) = _problem("hetero")
    blr.post(hyp + 1.0, X, y, var_X)  # an earlier m that must not be kept

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(linalg, "cho_factor", fail)
    blr.post(hyp, X, y, var_X)
    np.testing.assert_allclose(blr.m, _reference_m(blr, X, y), rtol=1e-10)
