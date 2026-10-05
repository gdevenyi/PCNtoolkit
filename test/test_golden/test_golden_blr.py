"""Golden-value tests for BLR.

Three tolerance situations:

- Fixed hyperparameters (no optimiser) and predictions from a saved model are
  deterministic: DETERMINISTIC class, rtol 1e-10.
- A re-fit runs the l-bfgs-b optimiser, whose path a later PR may change:
  OPTIMISER class, converged negative log-likelihood rtol 1e-8 and Z-scores
  atol 1e-6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from test.test_golden._common import (
    BLR_CONFIGS,
    OPT_NLL_RTOL,
    OPT_Z_ATOL,
    RESPONSE_VAR,
    assert_deterministic,
    blr_fixed_hyp_outputs,
    blr_template,
    copy_model,
    load_npz,
    make_normative_model,
    predict_outputs,
    sampling_outputs,
    to_normdata,
)

GOLDEN: dict[str, np.ndarray] = load_npz("blr")
PREDICT_KEYS: tuple[str, ...] = ("Z", "centiles", "logp", "yhat")


def _fixed_inputs() -> dict[str, np.ndarray]:
    """Return the stored standardised inputs for the fixed-hyp evaluations."""
    names = ("X_train", "be_train", "Y_train", "X_test", "be_test", "Y_test", "Z_test")
    return {n: GOLDEN[f"fx_{n}"] for n in names}


def _test_data() -> NormData:
    """Return a fresh NormData of the held-out BLR data."""
    return to_normdata("test", GOLDEN["test_X"], GOLDEN["test_be"], GOLDEN["test_Y"])


@pytest.fixture(scope="module", params=BLR_CONFIGS)
def fixed_hyp(request: pytest.FixtureRequest) -> tuple[str, dict[str, np.ndarray]]:
    """Evaluate one BLR config at its stored fixed hyperparameters."""
    config: str = request.param
    return config, blr_fixed_hyp_outputs(
        config, GOLDEN[f"fx_{config}_hyp"], **_fixed_inputs()
    )


@pytest.fixture(scope="module", params=BLR_CONFIGS)
def saved_predictions(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, dict[str, np.ndarray]]:
    """Predict the held-out data with one saved golden BLR model."""
    config: str = request.param
    model = copy_model(f"blr_{config}", tmp_path_factory.mktemp(f"blr_{config}"))
    return config, predict_outputs(model, _test_data())


@pytest.fixture(scope="module", params=BLR_CONFIGS)
def refit(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, float, np.ndarray]:
    """Re-fit one BLR config on the stored training data."""
    config: str = request.param
    save_dir = tmp_path_factory.mktemp(f"refit_{config}")
    model = make_normative_model(blr_template(config), str(save_dir))
    model.fit(
        to_normdata("train", GOLDEN["train_X"], GOLDEN["train_be"], GOLDEN["train_Y"])
    )
    data = _test_data()
    model.compute_zscores(data)
    return (
        config,
        float(model.regression_models[RESPONSE_VAR].nlZ),
        data["Z"].values.copy(),
    )


def test_013_blrLoglik_should_matchGolden_when_givenFixedHyperparameters(
    fixed_hyp: tuple[str, dict[str, np.ndarray]],
) -> None:
    """
    Arrange: standardised training data and a fixed hyperparameter vector.
    Act: loglik and penalized_loglik (L1 and L2).
    Assert: equal to the golden values.
    """
    config, actual = fixed_hyp
    for key in ("nll", "pnll_l1", "pnll_l2"):
        assert_deterministic(
            actual[key], GOLDEN[f"fx_{config}_{key}"], name=f"{config}/{key}"
        )


def test_014_blrPost_should_matchGolden_when_givenFixedHyperparameters(
    fixed_hyp: tuple[str, dict[str, np.ndarray]],
) -> None:
    """
    Arrange: as test 013.
    Act: post (run inside loglik).
    Assert: posterior mean m and precision A equal the golden values
        (m has signed entries near 0; A has off-diagonal zeros).
    """
    config, actual = fixed_hyp
    for key in ("m", "A"):
        assert_deterministic(
            actual[key],
            GOLDEN[f"fx_{config}_{key}"],
            near_zero=True,
            name=f"{config}/{key}",
        )


def test_015_blrDloglik_should_matchGolden_when_givenFixedHyperparameters() -> None:
    """
    Arrange: the plain config (dloglik supports neither warps nor
        heteroskedastic noise) and its fixed hyperparameters.
    Act: dloglik.
    Assert: equal to the golden gradient (signed, so atol is on).
    """
    actual = blr_fixed_hyp_outputs("plain", GOLDEN["fx_plain_hyp"], **_fixed_inputs())
    assert_deterministic(
        actual["dnll"], GOLDEN["fx_plain_dnll"], near_zero=True, name="dnll"
    )


def test_016_blrYsS2_should_matchGolden_when_givenFixedHyperparameters(
    fixed_hyp: tuple[str, dict[str, np.ndarray]],
) -> None:
    """
    Arrange: as test 013, plus held-out covariates.
    Act: ys_s2.
    Assert: predictive mean (signed) and variance equal the golden values.
    """
    config, actual = fixed_hyp
    assert_deterministic(
        actual["ys"], GOLDEN[f"fx_{config}_ys"], near_zero=True, name=f"{config}/ys"
    )
    assert_deterministic(actual["s2"], GOLDEN[f"fx_{config}_s2"], name=f"{config}/s2")


@pytest.mark.parametrize("key", ["forward", "backward", "logp"])
def test_017_blrMaps_should_matchGolden_when_givenFixedHyperparameters(
    fixed_hyp: tuple[str, dict[str, np.ndarray]], key: str
) -> None:
    """
    Arrange: as test 016, plus held-out Y and Z.
    Act: forward, backward and elemwise_logp.
    Assert: equal to the golden values (all signed, so atol is on).
    """
    config, actual = fixed_hyp
    assert_deterministic(
        actual[key],
        GOLDEN[f"fx_{config}_{key}"],
        near_zero=True,
        name=f"{config}/{key}",
    )


@pytest.mark.parametrize("key", PREDICT_KEYS)
def test_018_blrSavedModel_should_reproducePredictions_when_loaded(
    saved_predictions: tuple[str, dict[str, np.ndarray]], key: str
) -> None:
    """
    Arrange: a saved golden BLR model (plain, warp, heteroskedastic), copied
        to a temporary directory.
    Act: compute_zscores, compute_centiles, compute_logp, compute_yhat on the
        held-out data.
    Assert: equal to the golden values (all signed, so atol is on).
    """
    config, actual = saved_predictions
    assert_deterministic(
        actual[key],
        GOLDEN[f"pred_{config}_{key}"],
        near_zero=True,
        name=f"{config}/{key}",
    )


def test_019_blrRefit_should_convergeToGoldenNll_when_fittedOnStoredData(
    refit: tuple[str, float, np.ndarray],
) -> None:
    """
    Arrange: the stored training data and the BLR template.
    Act: NormativeModel.fit (l-bfgs-b).
    Assert: converged negative log-likelihood within rtol 1e-8 of golden.
    """
    config, nlZ, _ = refit
    np.testing.assert_allclose(
        nlZ, GOLDEN[f"fit_{config}_nlZ"], rtol=OPT_NLL_RTOL, err_msg=config
    )


def test_020_blrRefit_should_giveGoldenZscores_when_fittedOnStoredData(
    refit: tuple[str, float, np.ndarray],
) -> None:
    """
    Arrange: as test 019.
    Act: compute_zscores on the held-out data with the re-fitted model.
    Assert: Z-scores within atol 1e-6 of the golden saved-model Z-scores.
    """
    config, _, Z = refit
    np.testing.assert_allclose(
        Z, GOLDEN[f"pred_{config}_Z"], rtol=0, atol=OPT_Z_ATOL, err_msg=config
    )


def test_021_sampleCovariates_should_matchGolden_when_globalSeedIsFixed(
    tmp_path: Path,
) -> None:
    """
    Arrange: the saved plain BLR model and the stored seed.
    Act: sample_batch_effects, then sample_covariates in both modes, with
        np.random seeded (both use NumPy's global random state).
    Assert: the same batch effects, and covariates equal to the golden values.
    """
    model = copy_model("blr_plain", tmp_path)
    actual = sampling_outputs(
        model, int(GOLDEN["plain_sampling_seed"]), GOLDEN["plain_sample_bes"].shape[0]
    )
    np.testing.assert_array_equal(actual["sample_bes"], GOLDEN["plain_sample_bes"])
    # Covariates are drawn in raw units (age 8-85, cov2 around 0).
    for key in ("sample_X", "sample_X_per_be"):
        assert_deterministic(
            actual[key], GOLDEN[f"plain_{key}"], near_zero=True, name=key
        )
