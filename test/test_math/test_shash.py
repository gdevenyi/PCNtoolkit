"""Tests for the deduplication in the shash helpers K, P and m1m2.

The helpers compute the Bessel function once per repeated input value. The
results must be bit-identical to a direct computation on every element.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.special as spp

from pcntoolkit.math_functions import shash

N, S, LEVELS = 300, 40, 5
FRAC = np.exp(1 / 4) / np.sqrt(8 * np.pi)


@pytest.fixture(autouse=True)
def small_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use small sample and chunk sizes, so small arrays use every code path."""
    monkeypatch.setattr(shash, "_UNIQUE_SAMPLE_SIZE", 1000)
    monkeypatch.setattr(shash, "_CHUNK", 2000)


def _delta(case: str) -> np.ndarray:
    """Delta arrays of shape (observations, samples), as in HBR predict."""
    rng = np.random.default_rng(0)
    if case == "fixed":
        return np.repeat(rng.uniform(0.5, 2, (1, S)), N, axis=0)
    if case == "random_effect":
        return rng.uniform(0.5, 2, (LEVELS, S))[rng.integers(0, LEVELS, N)]
    if case == "linear":
        return rng.uniform(0.5, 2, (N, S))
    if case == "special":
        return np.array([[np.nan, 1.0, np.inf], [1.0, 0.7, np.nan]])
    if case == "empty":
        return np.empty((0, S))
    if case == "3d":
        return np.repeat(rng.uniform(0.5, 2, (4, 1, 6)), 3, axis=1)
    raise ValueError(case)


def _direct_P(q: np.ndarray) -> np.ndarray:
    return (spp.kv((q + 1) / 2, 1 / 4) + spp.kv((q - 1) / 2, 1 / 4)) * FRAC


CASES = ["fixed", "random_effect", "linear", "special", "empty", "3d"]


@pytest.mark.parametrize("case", CASES)
def test_shashHelpers_should_beBitIdentical_when_inputHasRepeatedValues(
    case: str,
) -> None:
    """
    Arrange: delta with fixed, per-level, distinct, special or no values.
    Act: K, P and m1m2 with deduplication.
    Assert: the results equal a direct computation on every element.
    """
    delta = _delta(case)
    epsilon = np.sin(np.arange(delta.size)).reshape(delta.shape)
    np.testing.assert_array_equal(shash.K(delta, 0.25), spp.kv(delta, 0.25))
    np.testing.assert_array_equal(shash.P(delta), _direct_P(delta))
    inv_delta = 1.0 / delta
    m1 = np.sinh(epsilon / delta) * _direct_P(inv_delta)
    m2 = (np.cosh(2 * (epsilon / delta)) * _direct_P(2.0 * inv_delta) - 1) / 2
    mean, raw_second = shash.m1m2(epsilon, delta)
    np.testing.assert_array_equal(mean, m1)
    np.testing.assert_array_equal(raw_second, m2)


@pytest.mark.parametrize(
    ("case", "n_reduced"),
    [("fixed", S), ("random_effect", LEVELS * S), ("linear", N * S)],
)
def test_dedupe_should_reduceToDistinctValues_when_valuesRepeat(
    case: str, n_reduced: int
) -> None:
    """
    Arrange: delta that is fixed per sample, per level, or all distinct.
    Act: _dedupe, then expand the reduced values.
    Assert: the reduced size is as expected, and expand restores the input.
    """
    delta = _delta(case)
    reduced, expand = shash._dedupe(delta)
    assert reduced.size == n_reduced
    restored = expand(reduced)
    np.testing.assert_array_equal(restored, delta)
    assert restored.flags.writeable


def test_shashHelpers_should_keepScalarTypes_when_givenScalars() -> None:
    """
    Arrange: scalar epsilon and delta.
    Act: K, P and m1m2.
    Assert: the results are numpy scalars equal to the direct computation.
    """
    assert shash.K(1.5, 0.25) == spp.kv(1.5, 0.25)
    assert shash.P(1.5) == _direct_P(1.5)
    mean, raw_second = shash.m1m2(0.3, 1.2)
    assert np.ndim(mean) == 0 and np.ndim(raw_second) == 0
    assert mean == np.sinh(0.3 / 1.2) * _direct_P(1.0 / 1.2)


@pytest.mark.parametrize("cpus", [1, 3])
def test_elementwise_should_useAtMostAllocatedThreads_when_arraysAreLarge(
    monkeypatch: pytest.MonkeyPatch, cpus: int
) -> None:
    """
    Arrange: distinct (linear) delta that spans several chunks; 1 or 3
        allocated CPUs; a spy on the thread pool.
    Act: m1m2 and P.
    Assert: no pool for 1 CPU, a pool of at most 3 threads otherwise, and
        results bit-identical to a direct computation.
    """
    pools: list[int] = []
    real_pool = shash.ThreadPoolExecutor

    def spy(max_workers: int):
        pools.append(max_workers)
        return real_pool(max_workers=max_workers)

    monkeypatch.setattr(shash, "allocated_cpus", lambda: cpus)
    monkeypatch.setattr(shash, "ThreadPoolExecutor", spy)
    delta = _delta("linear")
    epsilon = np.sin(np.arange(delta.size)).reshape(delta.shape)
    np.testing.assert_array_equal(shash.P(delta), _direct_P(delta))
    mean, raw_second = shash.m1m2(epsilon, delta)
    inv_delta = 1.0 / delta
    np.testing.assert_array_equal(mean, np.sinh(epsilon / delta) * _direct_P(inv_delta))
    expected = (np.cosh(2 * (epsilon / delta)) * _direct_P(2.0 * inv_delta) - 1) / 2
    np.testing.assert_array_equal(raw_second, expected)
    assert pools == ([] if cpus == 1 else [3, 3])
