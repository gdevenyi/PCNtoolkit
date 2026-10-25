import numpy as np
import pytest

from pcntoolkit.util import data_utils
from pcntoolkit.util.data_utils import clear_dedupe_cache, dedupe, elementwise


@pytest.fixture(autouse=True)
def empty_cache():
    clear_dedupe_cache()
    yield
    clear_dedupe_cache()


def _roundtrip(q: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    reduced, expand = dedupe(q, **kwargs)
    return reduced, expand(np.sqrt(reduced))


def test_dedupe_constant_axis():
    rng = np.random.default_rng(0)
    q = np.broadcast_to(rng.uniform(1, 2, (1, 30)), (50, 30)).copy()
    reduced, out = _roundtrip(q)
    assert reduced.size == 30
    assert np.array_equal(out, np.sqrt(q))


def test_dedupe_few_distinct():
    rng = np.random.default_rng(0)
    q = rng.uniform(1, 2, (5, 30))[rng.integers(0, 5, 200)]
    reduced, out = _roundtrip(q, sample_size=1000)
    assert reduced.size == 150
    assert np.array_equal(out, np.sqrt(q))


def test_dedupe_all_distinct_skips_unique():
    q = np.random.default_rng(0).uniform(1, 2, (200, 30))
    reduced, out = _roundtrip(q, sample_size=1000)
    assert reduced is q
    assert np.array_equal(out, np.sqrt(q))
    assert len(data_utils._unique_cache) == 0


def test_dedupe_empty_and_small():
    for q in [np.zeros((0, 3)), np.array([1.0, 2.0])]:
        reduced, out = _roundtrip(q)
        assert np.array_equal(out, np.sqrt(q))


def test_dedupe_cache_reuses_unique():
    rng = np.random.default_rng(0)
    q = rng.uniform(1, 2, (5, 30))[rng.integers(0, 5, 200)]
    first, _ = dedupe(q, sample_size=1000)
    second, out = dedupe(q.copy(), sample_size=1000)
    assert second is first
    assert np.array_equal(out(np.sqrt(second)), np.sqrt(q))


def test_dedupe_cache_checks_content():
    # Same shape and same sample, but other values: the cache must not be used.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 5, 200)
    q1 = rng.uniform(1, 2, (5, 30))[idx]
    q2 = q1.copy()
    q2[1, 1] = 9.0
    step = q1.size // 1000
    assert np.array_equal(q1.reshape(-1)[::step], q2.reshape(-1)[::step])
    dedupe(q1, sample_size=1000)
    reduced, out = _roundtrip(q2, sample_size=1000)
    assert 9.0 in reduced
    assert np.array_equal(out, np.sqrt(q2))


def test_dedupe_cache_size():
    rng = np.random.default_rng(0)
    for i in range(4):
        q = rng.uniform(1, 2, (5, 30))[rng.integers(0, 5, 200)] + i
        dedupe(q, sample_size=1000, cache_size=2)
    assert len(data_utils._unique_cache) == 2
    dedupe(q, sample_size=1000, cache_size=0)
    clear_dedupe_cache()
    dedupe(q, sample_size=1000, cache_size=0)
    assert len(data_utils._unique_cache) == 0


def test_elementwise_chunks():
    a = np.random.default_rng(0).uniform(1, 2, (40, 30))
    b = a[:10]
    out_a, out_b = elementwise(np.sqrt, a, b, chunk_size=100)
    assert np.array_equal(out_a, np.sqrt(a))
    assert np.array_equal(out_b, np.sqrt(b))
