"""Golden-value tests for ``NormativeModel(n_jobs=2)``.

With ``n_jobs=2`` the response variables run in two worker processes with
1 BLAS thread each. The results must stay in the same tolerance classes as
with ``n_jobs=1``: predictions from a saved model DETERMINISTIC, a BLR re-fit
OPTIMISER (converged negative log-likelihood rtol 1e-8).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.util import parallel
from test.test_golden._common import (
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


@pytest.fixture(scope="module")
def two_workers() -> Iterator[list[int]]:
    """Allow 2 workers whatever the machine, and count the Parallel runs."""
    calls: list[int] = []
    real_parallel = parallel.Parallel

    def counting_parallel(*args, **kwargs):
        calls.append(kwargs["n_jobs"])
        return real_parallel(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(parallel, "cpu_count", lambda only_physical_cores=False: 2)
        for name in parallel.ALLOCATION_ENV_VARS:
            mp.delenv(name, raising=False)
        mp.setattr(parallel, "Parallel", counting_parallel)
        yield calls


@pytest.fixture(scope="module")
def blr_predictions(
    tmp_path_factory: pytest.TempPathFactory, two_workers: list[int]
) -> dict[str, dict[str, np.ndarray]]:
    """Predict the held-out data with the saved BLR multi model, n_jobs=2."""
    n_before = len(two_workers)
    model = copy_model("blr_multi", tmp_path_factory.mktemp("njobs_blr"))
    model.n_jobs = 2
    data = to_normdata(
        "test",
        GOLDEN["test_X"],
        GOLDEN["test_be"],
        GOLDEN["test_Y"],
        response_vars=RESPONSE_VARS,
    )
    outputs = predict_outputs_by_name(model, data)
    # One worker pool per compute_* call (Z, centiles, logp, yhat).
    assert two_workers[n_before:] == [2, 2, 2, 2]
    return outputs


@pytest.mark.parametrize("key", PREDICT_KEYS)
def test_035_blrPredict_should_matchGolden_when_nJobsIsTwo(
    blr_predictions: dict[str, dict[str, np.ndarray]], key: str
) -> None:
    """
    Arrange: the saved BLR model with three response variables, n_jobs=2.
    Act: compute_zscores, compute_centiles, compute_logp, compute_yhat.
    Assert: each response variable's values equal its golden values.
    """
    assert set(blr_predictions[key]) == set(RESPONSE_VARS)
    for rv in RESPONSE_VARS:
        assert_deterministic(
            blr_predictions[key][rv],
            GOLDEN[f"blr_pred_{key}__{rv}"],
            near_zero=True,
            name=f"blr/{key}/{rv}",
        )


def test_036_blrRefit_should_convergeToGoldenNll_when_nJobsIsTwo(
    tmp_path_factory: pytest.TempPathFactory, two_workers: list[int]
) -> None:
    """
    Arrange: the stored training data with three response variables, the
        plain BLR template and n_jobs=2.
    Act: NormativeModel.fit.
    Assert: each fitted model is back in the parent process and its
        converged negative log-likelihood is within rtol 1e-8 of golden.
    """
    model = make_normative_model(
        blr_template("plain"), str(tmp_path_factory.mktemp("njobs_refit"))
    )
    model.n_jobs = 2
    model.fit(
        to_normdata(
            "train",
            GOLDEN["train_X"],
            GOLDEN["train_be"],
            GOLDEN["train_Y"],
            response_vars=RESPONSE_VARS,
        )
    )
    assert two_workers, "the worker path did not run"
    assert set(model.regression_models) == set(RESPONSE_VARS)
    for rv in RESPONSE_VARS:
        assert model.regression_models[rv].is_fitted, rv
        np.testing.assert_allclose(
            float(model.regression_models[rv].nlZ),
            GOLDEN[f"blr_fit_nlZ__{rv}"],
            rtol=OPT_NLL_RTOL,
            err_msg=rv,
        )


def test_037_hbrModel_should_runSequentially_when_nJobsIsTwo(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """
    Arrange: the saved HBR multi model with n_jobs=2.
    Act: ask for the number of workers.
    Assert: a warning, and 1 worker (HBR uses ``cores`` for its chains).
    """
    model = copy_model("hbr_multi", tmp_path_factory.mktemp("njobs_hbr"))
    model.n_jobs = 2
    with pytest.warns(UserWarning, match="ignored for HBR"):
        assert model._n_workers() == 1


@pytest.mark.parametrize("n_jobs", [0, 2.0, None])
def test_038_normativeModel_should_raise_when_nJobsInvalid(n_jobs: object) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        NormativeModel(blr_template("plain"), n_jobs=n_jobs)  # type: ignore[arg-type]
