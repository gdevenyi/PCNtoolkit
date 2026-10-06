"""Golden-value tests for models with several response variables.

Every output is matched to its response variable by name, never by position.
``NormativeModel.load`` reads the response variables in ``glob`` order and
the ``compute_*`` methods loop over a ``set``, so the order can change between
processes. A change of order alone must not fail these tests; an output that
belongs to the wrong response variable must.

Predictions from saved models: DETERMINISTIC class. BLR re-fit: OPTIMISER
class (converged negative log-likelihood rtol 1e-8).
"""

from __future__ import annotations

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from test.test_golden._common import (
    N_MULTI_RESPONSE_VARS,
    OPT_NLL_RTOL,
    assert_deterministic,
    blr_template,
    copy_model,
    load_npz,
    make_normative_model,
    predict_outputs_by_name,
    to_normdata,
)

GOLDEN: dict[str, np.ndarray] = load_npz("multi")
RESPONSE_VARS: list[str] = [str(rv) for rv in GOLDEN["response_vars"]]
PREDICT_KEYS: tuple[str, ...] = ("Z", "centiles", "logp", "yhat")
MODEL_KINDS: tuple[str, ...] = ("blr", "hbr")
# Column orders of the held-out data: as stored, and reversed.
ORDERS: dict[str, list[int]] = {
    "stored": list(range(N_MULTI_RESPONSE_VARS)),
    "reversed": list(reversed(range(N_MULTI_RESPONSE_VARS))),
}


def _test_data(order: list[int]) -> NormData:
    """Return the held-out data with the response variables in ``order``.

    Parameters
    ----------
    order : list[int]
        Column indices into the stored ``test_Y``.

    Returns
    -------
    NormData
        Fresh data; each column keeps its own name.
    """
    return to_normdata(
        "test",
        GOLDEN["test_X"],
        GOLDEN["test_be"],
        GOLDEN["test_Y"][:, order],
        response_vars=[RESPONSE_VARS[i] for i in order],
    )


CASES: list[tuple[str, str]] = [(k, o) for k in MODEL_KINDS for o in ORDERS]


@pytest.fixture(scope="module", params=CASES, ids=[f"{k}-{o}" for k, o in CASES])
def predictions(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, dict[str, dict[str, np.ndarray]]]:
    """Predict the held-out data with one saved multi-response model."""
    kind, order = request.param
    model = copy_model(f"{kind}_multi", tmp_path_factory.mktemp(f"{kind}_{order}"))
    return kind, predict_outputs_by_name(model, _test_data(ORDERS[order]))


def test_027_multiGolden_should_holdThreeResponseVars_when_loaded() -> None:
    """
    Arrange: the golden multi-response file.
    Act: list the response variables and the pinned keys.
    Assert: three distinct names, and every model kind has every prediction
        output and (BLR) a fit result for each name.
    """
    assert len(set(RESPONSE_VARS)) == N_MULTI_RESPONSE_VARS
    for rv in RESPONSE_VARS:
        assert f"blr_fit_nlZ__{rv}" in GOLDEN
        for kind in MODEL_KINDS:
            for key in PREDICT_KEYS:
                assert f"{kind}_pred_{key}__{rv}" in GOLDEN


@pytest.mark.parametrize("key", PREDICT_KEYS)
def test_028_multiSavedModel_should_matchGoldenByName_when_orderVaries(
    predictions: tuple[str, dict[str, dict[str, np.ndarray]]], key: str
) -> None:
    """
    Arrange: a saved BLR or HBR model with three response variables, and the
        held-out data with its columns in stored or reversed order.
    Act: compute_zscores, compute_centiles, compute_logp, compute_yhat.
    Assert: the same set of response variable names as golden, and each
        name's values equal that name's golden values (signed, so atol is on).
    """
    kind, actual = predictions
    assert set(actual[key]) == set(RESPONSE_VARS), kind
    for rv in RESPONSE_VARS:
        assert_deterministic(
            actual[key][rv],
            GOLDEN[f"{kind}_pred_{key}__{rv}"],
            near_zero=True,
            name=f"{kind}/{key}/{rv}",
        )


def test_029_multiBlrRefit_should_convergeToGoldenNllByName_when_fitted(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """
    Arrange: the stored training data with three response variables, and
        the plain BLR template.
    Act: NormativeModel.fit (one l-bfgs-b fit per response variable).
    Assert: the same set of fitted names as golden, and each name's
        converged negative log-likelihood within rtol 1e-8 of its golden value.
    """
    model = make_normative_model(
        blr_template("plain"), str(tmp_path_factory.mktemp("multi_refit"))
    )
    model.fit(
        to_normdata(
            "train",
            GOLDEN["train_X"],
            GOLDEN["train_be"],
            GOLDEN["train_Y"],
            response_vars=RESPONSE_VARS,
        )
    )
    assert set(model.regression_models) == set(RESPONSE_VARS)
    for rv in RESPONSE_VARS:
        np.testing.assert_allclose(
            float(model.regression_models[rv].nlZ),
            GOLDEN[f"blr_fit_nlZ__{rv}"],
            rtol=OPT_NLL_RTOL,
            err_msg=rv,
        )
