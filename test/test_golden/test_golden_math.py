"""Golden-value tests for shash helpers, likelihood numpy maps and warps.

All functions here are deterministic, so they use the DETERMINISTIC tolerance
(rtol 1e-10). Values that can be close to 0 also get atol 1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest

from test.test_golden._common import (
    SHASH_CHUNK,
    SHASH_CHUNK_SHAPES,
    SHASH_SUMMARY_SHAPES,
    assert_deterministic,
    likelihood_outputs,
    load_npz,
    shash_chunk_outputs,
    shash_outputs,
    warp_outputs,
)

GOLDEN: dict[str, np.ndarray] = load_npz("math")
CHUNK_GOLDEN: dict[str, np.ndarray] = load_npz("shash_chunks")

LIKELIHOOD_KEYS: list[str] = sorted(
    k
    for k in GOLDEN
    if k.startswith("lik_") and k.rsplit("_", 1)[-1] in ("forward", "backward", "yhat")
)
WARP_KEYS: list[str] = sorted(
    k
    for k in GOLDEN
    if k.startswith("warp_") and k.rsplit("_", 1)[-1] in ("f", "invf", "df")
)


@pytest.fixture(scope="module")
def shash_actual() -> dict[str, np.ndarray]:
    """Recompute the shash outputs once per module."""
    return shash_outputs(GOLDEN)


@pytest.fixture(scope="module")
def shash_chunk_actual() -> dict[str, np.ndarray]:
    """Recompute the multi-chunk shash outputs once per module."""
    return shash_chunk_outputs()


@pytest.fixture(scope="module")
def likelihood_actual() -> dict[str, np.ndarray]:
    """Recompute the likelihood outputs once per module."""
    return likelihood_outputs(GOLDEN)


@pytest.fixture(scope="module")
def warp_actual() -> dict[str, np.ndarray]:
    """Recompute the warp outputs once per module."""
    return warp_outputs(GOLDEN)


@pytest.mark.parametrize("key", ["shash_S", "shash_S_inv"])
def test_001_shashTransform_should_matchGolden_when_givenFixedGrid(
    shash_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: a fixed grid of x, epsilon and delta.
    Act: evaluate S and S_inv.
    Assert: equal to the golden values (outputs cross 0, so atol is on).
    """
    assert_deterministic(shash_actual[key], GOLDEN[key], near_zero=True, name=key)


@pytest.mark.parametrize("key", ["shash_K_scalar", "shash_K_array", "shash_P"])
def test_002_shashBessel_should_matchGolden_when_givenFixedOrders(
    shash_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: fixed Bessel orders and P arguments.
    Act: evaluate K (float path and dask array path) and P.
    Assert: equal to the golden values.
    """
    assert_deterministic(shash_actual[key], GOLDEN[key], name=key)


@pytest.mark.parametrize("key", ["shash_m1", "shash_m2", "shash_m1m2_scalar"])
def test_003_shashMoments_should_matchGolden_when_givenFixedEpsilonDelta(
    shash_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: a fixed 2-D grid of epsilon and delta.
    Act: evaluate m1m2 on arrays and on scalars.
    Assert: equal to the golden values (m1 is ~0 when epsilon is ~0).
    """
    assert_deterministic(shash_actual[key], GOLDEN[key], near_zero=True, name=key)


@pytest.mark.parametrize("key", LIKELIHOOD_KEYS)
def test_004_likelihoodNumpyMap_should_matchGolden_when_givenFixedParameters(
    likelihood_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: fixed (observations, sample) parameter arrays for Normal, SHASHb,
        SHASHo, SHASHo2, Beta and ZINB; ZINB forward gets a seeded rng.
    Act: evaluate forward, backward and yhat.
    Assert: equal to the golden values (Z and Y cross 0, so atol is on).
    """
    assert_deterministic(likelihood_actual[key], GOLDEN[key], near_zero=True, name=key)


@pytest.mark.parametrize("key", WARP_KEYS)
def test_005_warp_should_matchGolden_when_givenFixedParameters(
    warp_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: positive x values and fixed parameters per warp.
    Act: evaluate f, invf(f(x)) and df.
    Assert: equal to the golden values (log-type warps cross 0 at x=1).
    """
    assert_deterministic(warp_actual[key], GOLDEN[key], near_zero=True, name=key)


def test_025_shashChunkGolden_should_coverEveryShapeAndFunction_when_loaded() -> None:
    """
    Arrange: the golden multi-chunk shash file and the input shapes.
    Act: list the pinned keys.
    Assert: every shape spans more than one dask chunk in at least one
        dimension (the "big" shape in both), and every shape has K, P, m1 and
        m2 (in full, or as a summary), so no case is dropped silently.
    """
    for name, shape in SHASH_CHUNK_SHAPES.items():
        assert max(shape) > SHASH_CHUNK, name
        if name in SHASH_SUMMARY_SHAPES:
            assert min(shape) > SHASH_CHUNK, name
            suffixes = ("_row_sumsq", "_col_sumsq", "_subset")
        else:
            suffixes = ("",)
        for func in ("K", "P", "m1", "m2"):
            for suffix in suffixes:
                assert f"chunk_{name}_{func}{suffix}" in CHUNK_GOLDEN


@pytest.mark.parametrize("key", sorted(CHUNK_GOLDEN))
def test_026_shashChunked_should_matchGolden_when_inputSpansSeveralChunks(
    shash_chunk_actual: dict[str, np.ndarray], key: str
) -> None:
    """
    Arrange: closed-form inputs of shape (1500, 3), (3, 1500) and
        (1100, 1050), larger than the (1000, 1000) dask chunks of P.
    Act: evaluate K (with P's chunks), P and m1m2.
    Assert: equal to the golden values; for the largest shape, row and
        column sums of squares and a fixed subset of values (m1 is signed
        and ~0 when epsilon is ~0, so atol is on for it).
    """
    assert_deterministic(
        shash_chunk_actual[key],
        CHUNK_GOLDEN[key],
        near_zero="_m1" in key,
        name=key,
    )
