"""Golden-value tests for HBR predictions from saved models.

The tests never sample. Each saved model holds a fixed posterior, and
prediction from it only evaluates deterministic functions of that posterior,
so the DETERMINISTIC tolerance applies.

ZINB Z-scores are the exception: ``ZeroInflatedNegativeBinomialLikelihood.forward``
draws random numbers and ``HBR.forward`` cannot pass it a seeded generator,
so they are only checked for shape and finiteness.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from test.test_golden._common import (
    HBR_LIKELIHOODS,
    assert_deterministic,
    copy_model,
    load_npz,
    predict_outputs,
    to_normdata,
)

GOLDEN: dict[str, np.ndarray] = load_npz("hbr")

CASES: list[tuple[str, str]] = [
    (name, key)
    for name in HBR_LIKELIHOODS
    for key in ("Z", "centiles", "logp", "yhat")
    if f"{name}_pred_{key}" in GOLDEN
]


def _test_data(name: str) -> NormData:
    """Return a fresh NormData of the held-out data for one likelihood."""
    return to_normdata(
        "test",
        GOLDEN[f"{name}_test_X"],
        GOLDEN[f"{name}_test_be"],
        GOLDEN[f"{name}_test_Y"],
    )


@pytest.fixture(scope="module")
def predictions(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, np.ndarray]]:
    """Predict the held-out data once with each saved HBR model."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in HBR_LIKELIHOODS:
        model = copy_model(f"hbr_{name}", tmp_path_factory.mktemp(f"hbr_{name}"))
        out[name] = predict_outputs(model, _test_data(name))
    return out


def test_022_hbrGoldenFiles_should_coverEveryLikelihood_when_loaded() -> None:
    """
    Arrange: the golden HBR file.
    Act: list the pinned prediction outputs.
    Assert: every likelihood has centiles, logp and yhat, and all but ZINB
        also have Z-scores, so no case is dropped silently.
    """
    for name in HBR_LIKELIHOODS:
        expected = {"centiles", "logp", "yhat"} | ({"Z"} if name != "ZINB" else set())
        assert {k for n, k in CASES if n == name} == expected, name


@pytest.mark.parametrize(("name", "key"), CASES, ids=[f"{n}-{k}" for n, k in CASES])
def test_023_hbrSavedModel_should_reproducePredictions_when_loaded(
    predictions: dict[str, dict[str, np.ndarray]], name: str, key: str
) -> None:
    """
    Arrange: a saved golden HBR model (Normal, SHASHb, Beta, ZINB), copied to
        a temporary directory.
    Act: compute_zscores, compute_centiles, compute_logp, compute_yhat on the
        held-out data.
    Assert: equal to the golden values. All are signed or can be ~0 (ZINB
        centiles are exactly 0 at low centiles), so atol is on; logp also
        varies in the last bit between processes.
    """
    assert_deterministic(
        predictions[name][key],
        GOLDEN[f"{name}_pred_{key}"],
        near_zero=True,
        name=f"{name}/{key}",
    )


def test_024_hbrZinbZscores_should_beFinite_when_predictedFromSavedModel(
    predictions: dict[str, dict[str, np.ndarray]],
) -> None:
    """
    Arrange: the saved ZINB model and its held-out count data.
    Act: compute_zscores (randomised quantile residuals, unseeded).
    Assert: one finite Z-score per subject. The values are random, so they
        are not compared with golden values.
    """
    Z = predictions["ZINB"]["Z"]
    assert Z.shape == GOLDEN["ZINB_pred_logp"].shape
    assert np.all(np.isfinite(Z))
