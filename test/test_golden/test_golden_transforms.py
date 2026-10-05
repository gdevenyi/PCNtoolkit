"""Golden-value tests for scalers and basis functions (DETERMINISTIC class)."""

from __future__ import annotations

import numpy as np
import pytest

from test.test_golden._common import (
    assert_deterministic,
    basis_outputs,
    load_npz,
    scaler_outputs,
)

GOLDEN: dict[str, np.ndarray] = load_npz("transforms")

SCALER_KEYS: list[str] = sorted(
    k
    for k in GOLDEN
    if k.startswith("scaler_") and k not in ("scaler_train", "scaler_test")
)
BASIS_KEYS: list[str] = sorted(
    k
    for k in GOLDEN
    if k.startswith("basis_") and k not in ("basis_train", "basis_test")
)


@pytest.fixture(scope="module")
def scaler_actual() -> dict[str, np.ndarray]:
    """Recompute the scaler outputs once per module."""
    return scaler_outputs(GOLDEN)


@pytest.fixture(scope="module")
def basis_actual() -> dict[str, np.ndarray]:
    """Recompute the basis function outputs once per module."""
    return basis_outputs(GOLDEN)


@pytest.mark.parametrize("key", SCALER_KEYS)
def test_006_scaler_should_matchGolden_when_fittedOnFixedData(
    scaler_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: fixed training and held-out matrices (held-out rows go past the
        training range).
    Act: fit each scaler, transform, inverse-transform, transform with index.
    Assert: equal to the golden values (standardised values cross 0).
    """
    assert_deterministic(scaler_actual[key], GOLDEN[key], near_zero=True, name=key)


@pytest.mark.parametrize("key", BASIS_KEYS)
def test_007_basisFunction_should_matchGolden_when_fittedOnFixedData(
    basis_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: fixed training and held-out covariates.
    Act: fit each basis function and transform the held-out data.
    Assert: equal to the golden values (B-spline columns are 0 outside their
        support, and the raw covariate column crosses 0).
    """
    assert_deterministic(basis_actual[key], GOLDEN[key], near_zero=True, name=key)
