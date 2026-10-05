"""Shared builders for the golden-value characterisation tests.

The generator (``test/resources/golden/generate_golden.py``) and the tests in
this package both import from here. The generator builds seeded inputs, runs
the functions below on them and saves inputs and outputs. The tests load the
saved inputs, run the same functions again and compare with the saved outputs.

The tests never draw new random numbers from a seed: they read every input
from the ``.npz`` files. A change in NumPy's random stream therefore cannot
break them.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.basis_function import (
    BsplineBasisFunction,
    LinearBasisFunction,
    PolynomialBasisFunction,
)
from pcntoolkit.math_functions.likelihood import (
    BetaLikelihood,
    Likelihood,
    NormalLikelihood,
    SHASHbLikelihood,
    SHASHo2Likelihood,
    SHASHoLikelihood,
    ZeroInflatedNegativeBinomialLikelihood,
)
from pcntoolkit.math_functions.prior import LinearPrior, Prior, RandomPrior, make_prior
from pcntoolkit.math_functions.scaler import Scaler
from pcntoolkit.math_functions.shash import K, P, S, S_inv, m1m2
from pcntoolkit.math_functions.warp import (
    WarpAffine,
    WarpBase,
    WarpBoxCox,
    WarpCompose,
    WarpLog,
    WarpSinhArcsinh,
)
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from pcntoolkit.regression_model.hbr import HBR
from pcntoolkit.util.data_utils import iter_batch_combinations
from pcntoolkit.util.evaluator import Evaluator

GOLDEN_DIR: Path = Path(__file__).parents[1] / "resources" / "golden"

# Master seed of the generator. Each section derives its own child seed.
SEED: int = 20261005
N_TRAIN: int = 200
N_TEST: int = 100

COVARIATES: list[str] = ["age", "cov2"]
BATCH_EFFECT_DIMS: list[str] = ["site", "sex"]
SITES: list[str] = ["s0", "s1", "s2"]
SEXES: list[str] = ["F", "M"]
RESPONSE_VAR: str = "y"

BLR_CONFIGS: tuple[str, ...] = ("plain", "warp", "hetero")
# SHASHo and SHASHo2 are left out: their Likelihood classes are abstract and
# cannot be instantiated, so no HBR model can use them (see README).
HBR_LIKELIHOODS: tuple[str, ...] = ("Normal", "SHASHb", "beta", "ZINB")

# HBR sampler settings. Small, because predict outputs only need a posterior,
# not a converged one.
HBR_DRAWS: int = 20
HBR_TUNE: int = 20
HBR_CHAINS: int = 2

# DETERMINISTIC tolerance class.
DET_RTOL: float = 1e-10
# Absolute floor for outputs that can be close to 0 (Z-scores, centred
# quantities). Without it, rtol on a value of 1e-9 asks for 1e-19 precision.
NEAR_ZERO_ATOL: float = 1e-12

# OPTIMISER tolerance class (BLR fit).
OPT_NLL_RTOL: float = 1e-8
OPT_Z_ATOL: float = 1e-6


# --------------------------------------------------------------------------- #
# Assertions and file helpers
# --------------------------------------------------------------------------- #


def assert_deterministic(
    actual: Any, expected: Any, near_zero: bool = False, name: str = ""
) -> None:
    """Compare an output with its golden value at the DETERMINISTIC tolerance.

    Parameters
    ----------
    actual : Any
        Recomputed value. Converted with ``np.asarray`` (also for dask arrays).
    expected : Any
        Golden value loaded from disk.
    near_zero : bool, optional
        True when the value can be close to 0; adds ``NEAR_ZERO_ATOL``.
    name : str, optional
        Name shown in the failure message.

    Raises
    ------
    AssertionError
        If the shapes differ or any element is outside the tolerance.
    """
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    assert actual_arr.shape == expected_arr.shape, (
        f"{name}: shape {actual_arr.shape} != {expected_arr.shape}"
    )
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        rtol=DET_RTOL,
        atol=NEAR_ZERO_ATOL if near_zero else 0.0,
        err_msg=name,
    )


def load_npz(name: str, golden_dir: Path = GOLDEN_DIR) -> dict[str, np.ndarray]:
    """Load one golden ``.npz`` file into a plain dict.

    Parameters
    ----------
    name : str
        File name without extension, e.g. ``"math"``.
    golden_dir : Path, optional
        Directory that holds the golden files.

    Returns
    -------
    dict[str, np.ndarray]
        All arrays in the file.

    Raises
    ------
    AssertionError
        If the file does not exist.
    """
    path = golden_dir / f"{name}.npz"
    assert path.exists(), (
        f"Missing golden file {path}. Run test/resources/golden/generate_golden.py."
    )
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def copy_model(name: str, dest: Path, golden_dir: Path = GOLDEN_DIR) -> NormativeModel:
    """Copy a saved golden model to ``dest`` and load it from there.

    Parameters
    ----------
    name : str
        Model directory name inside the golden directory.
    dest : Path
        Writable directory, normally pytest's ``tmp_path``.
    golden_dir : Path, optional
        Directory that holds the golden files.

    Returns
    -------
    NormativeModel
        The loaded model. Its ``save_dir`` is the copy, so the golden files
        stay untouched.
    """
    src = golden_dir / name
    assert (src / "model" / "normative_model.json").exists(), (
        f"Missing golden model {src}"
    )
    target = dest / name
    shutil.copytree(src, target)
    return NormativeModel.load(str(target))


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


def make_dataset_arrays(
    rng: np.random.Generator, n: int, kind: str
) -> dict[str, np.ndarray]:
    """Draw one synthetic normative dataset.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator.
    n : int
        Number of subjects.
    kind : str
        ``"gaussian"`` (real, skewed, heteroskedastic), ``"unit"`` (values in
        (0, 1), for the Beta likelihood) or ``"count"`` (zero-inflated counts,
        for the ZINB likelihood).

    Returns
    -------
    dict[str, np.ndarray]
        ``X`` (n, 2), ``be`` (n, 2) strings and ``Y`` (n, 1).

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    """
    age = rng.uniform(8.0, 85.0, n)
    cov2 = rng.normal(0.0, 1.0, n)
    site = rng.choice(SITES, n)
    sex = rng.choice(SEXES, n)
    site_shift = np.select([site == "s1", site == "s2"], [0.4, -0.3], 0.0)
    sex_shift = np.where(sex == "M", 0.2, 0.0)
    noise = rng.standard_normal(n)
    if kind == "gaussian":
        mean = 2.0 + 0.04 * age - 0.0004 * age**2 + 0.1 * cov2 + site_shift + sex_shift
        sd = 0.3 + 0.008 * age
        # Add a right skew so the warped and SHASH models have something to fit.
        y = mean + sd * (noise + 0.5 * rng.standard_exponential(n))
    elif kind == "unit":
        logit = -1.0 + 0.025 * age + 0.5 * site_shift + 0.4 * noise
        y = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98)
    elif kind == "count":
        lam = np.exp(0.5 + 0.015 * age + site_shift)
        y = rng.negative_binomial(2, 2.0 / (2.0 + lam)).astype(float)
        y[rng.uniform(size=n) < 0.25] = 0.0
    else:
        raise ValueError(f"Unknown dataset kind {kind}")
    return {
        "X": np.column_stack([age, cov2]),
        "be": np.column_stack([site, sex]),
        "Y": y[:, None],
    }


def to_normdata(name: str, X: np.ndarray, be: np.ndarray, Y: np.ndarray) -> NormData:
    """Build a NormData object with the fixed golden column names.

    Parameters
    ----------
    name : str
        Dataset name.
    X : np.ndarray
        Covariates, shape (n, 2).
    be : np.ndarray
        Batch effects as strings, shape (n, 2).
    Y : np.ndarray
        Responses, shape (n, n_response_vars).

    Returns
    -------
    NormData
        The dataset.
    """
    response_vars = (
        [RESPONSE_VAR]
        if Y.shape[1] == 1
        else [f"{RESPONSE_VAR}{i}" for i in range(Y.shape[1])]
    )
    attrs = {
        "covariates": COVARIATES,
        "batch_effect_dims": BATCH_EFFECT_DIMS,
        "response_vars": response_vars,
    }
    return NormData.from_ndarrays(name, X, Y, be, attrs=attrs)


# --------------------------------------------------------------------------- #
# Model templates
# --------------------------------------------------------------------------- #


def blr_template(config: str) -> BLR:
    """Build an unfitted BLR template.

    Parameters
    ----------
    config : str
        ``"plain"``, ``"warp"`` (SinhArcsinh warp) or ``"hetero"``
        (heteroskedastic noise).

    Returns
    -------
    BLR
        The template. All use the default l-bfgs-b optimiser.

    Raises
    ------
    ValueError
        If ``config`` is unknown.
    """
    bspline = BsplineBasisFunction(basis_column=0, nknots=5, degree=3)
    if config == "plain":
        return BLR(name="plain", fixed_effect=True, basis_function_mean=bspline)
    if config == "warp":
        return BLR(
            name="warp",
            fixed_effect=True,
            warp_name="WarpSinhArcsinh",
            basis_function_mean=bspline,
        )
    if config == "hetero":
        return BLR(
            name="hetero",
            fixed_effect=True,
            heteroskedastic=True,
            basis_function_mean=bspline,
            basis_function_var=LinearBasisFunction(basis_column=0),
        )
    raise ValueError(f"Unknown BLR config {config}")


def hbr_likelihood(name: str) -> Likelihood:
    """Build the likelihood (with priors) for one golden HBR model.

    Parameters
    ----------
    name : str
        One of ``HBR_LIKELIHOODS``.

    Returns
    -------
    Likelihood
        A fresh likelihood object.

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    if name == "Normal":
        mu = LinearPrior(
            slope=Prior(dist_name="Normal", dist_params=(0, 2)),
            intercept=RandomPrior(
                mu=Prior(dist_name="Normal", dist_params=(0, 2)),
                sigma=Prior(dist_name="HalfNormal", dist_params=(1.0,)),
            ),
            basis_function=BsplineBasisFunction(basis_column=0, nknots=4, degree=3),
        )
        sigma = LinearPrior(
            slope=Prior(dist_name="Normal", dist_params=(0, 1)),
            intercept=Prior(dist_name="Normal", dist_params=(1.0, 1.0)),
            mapping="softplus",
            mapping_params=(0, 3),
        )
        return NormalLikelihood(mu, sigma)
    if name == "SHASHb":
        mu = LinearPrior(
            slope=Prior(dist_name="Normal", dist_params=(0, 2)),
            intercept=RandomPrior(
                mu=Prior(dist_name="Normal", dist_params=(0, 2)),
                sigma=Prior(dist_name="HalfNormal", dist_params=(1.0,)),
            ),
        )
        sigma = Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
            mapping="softplus",
            mapping_params=(0, 3),
        )
        epsilon = Prior(dist_name="Normal", dist_params=(0.0, 1.0))
        delta = Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
            mapping="softplus",
            mapping_params=(0, 3, 0.6),
        )
        return SHASHbLikelihood(mu, sigma, epsilon, delta)
    if name == "beta":
        alpha = LinearPrior(
            slope=Prior(dist_name="Normal", dist_params=(0, 1)),
            intercept=Prior(dist_name="Normal", dist_params=(1.0, 1.0)),
            mapping="softplus",
            mapping_params=(0, 3),
        )
        beta = Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
            mapping="softplus",
            mapping_params=(0, 3),
        )
        return BetaLikelihood(alpha, beta)
    if name == "ZINB":
        mu = LinearPrior(
            slope=Prior(dist_name="Normal", dist_params=(0, 1)),
            intercept=Prior(dist_name="Normal", dist_params=(1.0, 1.0)),
            mapping="softplus",
            mapping_params=(0, 3),
        )
        alpha = Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
            mapping="softplus",
            mapping_params=(0, 3),
        )
        psi = Prior(
            dist_name="Normal",
            dist_params=(1.0, 1.0),
            mapping="sigmoid",
            mapping_params=(0, 1),
        )
        return ZeroInflatedNegativeBinomialLikelihood(mu, alpha, psi)
    raise ValueError(f"Unknown HBR likelihood {name}")


def hbr_dataset_kind(name: str) -> str:
    """Return the synthetic dataset kind that suits a likelihood.

    Parameters
    ----------
    name : str
        One of ``HBR_LIKELIHOODS``.

    Returns
    -------
    str
        ``"unit"`` for Beta, ``"count"`` for ZINB, else ``"gaussian"``.
    """
    return {"beta": "unit", "ZINB": "count"}.get(name, "gaussian")


def hbr_outscaler(name: str) -> str:
    """Return the outscaler that keeps Y inside the likelihood's support.

    Parameters
    ----------
    name : str
        One of ``HBR_LIKELIHOODS``.

    Returns
    -------
    str
        ``"none"`` for Beta and ZINB, else ``"standardize"``.
    """
    return "none" if name in ("beta", "ZINB") else "standardize"


def make_normative_model(
    template: BLR | HBR, save_dir: str, outscaler: str = "standardize"
) -> NormativeModel:
    """Wrap a template in a NormativeModel that writes no files on its own.

    Parameters
    ----------
    template : BLR | HBR
        Unfitted regression model template.
    save_dir : str
        Directory used if something does write (e.g. ``fit`` calls
        ``predict``, which creates the save sub-directories).
    outscaler : str, optional
        Response scaler name.

    Returns
    -------
    NormativeModel
        The meta-estimator.
    """
    return NormativeModel(
        template,
        savemodel=False,
        evaluate_model=False,
        saveresults=False,
        saveplots=False,
        save_dir=save_dir,
        inscaler="standardize",
        outscaler=outscaler,
    )


def predict_outputs(
    model: NormativeModel, data: NormData, zscores: bool = True
) -> dict[str, np.ndarray]:
    """Run the four prediction hot spots and collect their outputs.

    The calls run in the same order as ``NormativeModel.predict``.

    Parameters
    ----------
    model : NormativeModel
        A fitted model.
    data : NormData
        Held-out data. Modified in place.
    zscores : bool, optional
        Whether to call ``compute_zscores``.

    Returns
    -------
    dict[str, np.ndarray]
        ``Z``, ``centiles``, ``logp`` and ``yhat`` as plain arrays.
    """
    out: dict[str, np.ndarray] = {}
    if zscores:
        model.compute_zscores(data)
        out["Z"] = data["Z"].values.copy()
    model.compute_centiles(data)
    out["centiles"] = data["centiles"].values.copy()
    model.compute_logp(data)
    out["logp"] = data["logp"].values.copy()
    model.compute_yhat(data)
    out["yhat"] = data["Yhat"].values.copy()
    return out


# --------------------------------------------------------------------------- #
# BLR at fixed hyperparameters (pure functions of the inputs)
# --------------------------------------------------------------------------- #


def blr_be_maps() -> dict[str, dict[str, int]]:
    """Return the integer coding of the golden batch effects.

    Returns
    -------
    dict[str, dict[str, int]]
        Map from batch effect dimension to level to integer.
    """
    return {
        "site": {s: i for i, s in enumerate(SITES)},
        "sex": {s: i for i, s in enumerate(SEXES)},
    }


def encode_be(be: np.ndarray) -> np.ndarray:
    """Encode string batch effects as integers with ``blr_be_maps``.

    Parameters
    ----------
    be : np.ndarray
        Strings, shape (n, 2).

    Returns
    -------
    np.ndarray
        Integers, shape (n, 2).
    """
    maps = blr_be_maps()
    return np.column_stack(
        [[maps[d][v] for v in be[:, i]] for i, d in enumerate(BATCH_EFFECT_DIMS)]
    )


def blr_n_hyp(config: str, X: np.ndarray, be_int: np.ndarray) -> int:
    """Return the number of hyperparameters of a BLR config on given data.

    Parameters
    ----------
    config : str
        One of ``BLR_CONFIGS``.
    X : np.ndarray
        Covariates, shape (n, 2).
    be_int : np.ndarray
        Integer batch effects, shape (n, 2).

    Returns
    -------
    int
        Length of the hyperparameter vector.
    """
    blr = blr_template(config)
    blr.be_maps = blr_be_maps()
    Phi, Phi_var = blr.Phi_Phi_var(X, be_int)
    blr.D, blr.var_D = Phi.shape[1], Phi_var.shape[1]
    return int(blr.init_hyp().size)


def blr_fixed_hyp_outputs(
    config: str,
    hyp: np.ndarray,
    X_train: np.ndarray,
    be_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    be_test: np.ndarray,
    Y_test: np.ndarray,
    Z_test: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the BLR numeric core at a fixed hyperparameter vector.

    No optimiser runs, so all outputs are pure functions of the inputs.

    Parameters
    ----------
    config : str
        One of ``BLR_CONFIGS``.
    hyp : np.ndarray
        Fixed hyperparameters (log scale, as the optimiser sees them).
    X_train, be_train, Y_train : np.ndarray
        Training covariates, integer batch effects, responses (1-D).
    X_test, be_test, Y_test : np.ndarray
        Held-out covariates, integer batch effects, responses (1-D).
    Z_test : np.ndarray
        Z-values used for ``backward``, shape (n_test,).

    Returns
    -------
    dict[str, np.ndarray]
        ``nll``, ``pnll_l1``, ``pnll_l2``, ``m``, ``A``, ``ys``, ``s2``,
        ``forward``, ``backward``, ``logp`` and, for ``plain``, ``dnll``.
    """
    blr = blr_template(config)
    blr.be_maps = blr_be_maps()
    Phi, Phi_var = blr.Phi_Phi_var(X_train, be_train)
    blr.D, blr.var_D = Phi.shape[1], Phi_var.shape[1]
    out: dict[str, np.ndarray] = {}
    # loglik runs post() first, which sets m and A for this hyp.
    out["nll"] = np.asarray(blr.loglik(hyp, Phi, Y_train, Phi_var))
    out["m"] = blr.m.copy()
    out["A"] = blr.A.copy()
    out["pnll_l1"] = np.asarray(
        blr.penalized_loglik(hyp, Phi, Y_train, Phi_var, 0.1, "L1")
    )
    out["pnll_l2"] = np.asarray(
        blr.penalized_loglik(hyp, Phi, Y_train, Phi_var, 0.1, "L2")
    )
    if config == "plain":
        # A fresh model, so dloglik cannot reuse a cached posterior.
        fresh = blr_template(config)
        fresh.be_maps = blr_be_maps()
        fresh.D, fresh.var_D = blr.D, blr.var_D
        fresh.dnlZ = None
        out["dnll"] = fresh.dloglik(hyp, Phi, Y_train, Phi_var)
    _, blr.beta, blr.gamma = blr.parse_hyps(hyp, Phi, Phi_var)
    blr.is_fitted = True
    ys, s2 = blr.ys_s2(X_test, be_test)
    out["ys"], out["s2"] = ys.copy(), s2.copy()
    X_da = xr.DataArray(X_test, dims=("observations", "covariates"))
    be_da = xr.DataArray(be_test, dims=("observations", "batch_effect_dims"))
    Y_da = xr.DataArray(Y_test, dims=("observations",))
    Z_da = xr.DataArray(Z_test, dims=("observations",))
    out["forward"] = blr.forward(X_da, be_da, Y_da).values
    out["backward"] = blr.backward(X_da, be_da, Z_da).values
    out["logp"] = blr.elemwise_logp(X_da, be_da, Y_da).values
    return out


# --------------------------------------------------------------------------- #
# Pure math: shash, likelihood numpy maps, warps
# --------------------------------------------------------------------------- #


def shash_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate the shash helpers on a fixed grid.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``shash_x`` (1-D), ``shash_eps`` (1-D), ``shash_delta`` (1-D),
        ``shash_p`` (1-D, orders for ``K``), ``shash_q`` (2-D),
        ``shash_eps2d`` and ``shash_delta2d`` (2-D).

    Returns
    -------
    dict[str, np.ndarray]
        Outputs of ``S``, ``S_inv``, ``K`` (scalar and array path), ``P`` and
        ``m1m2`` (array and scalar path).
    """
    x = inp["shash_x"][:, None, None]
    e = inp["shash_eps"][None, :, None]
    d = inp["shash_delta"][None, None, :]
    out: dict[str, np.ndarray] = {
        "shash_S": S(x, e, d),
        "shash_S_inv": S_inv(x, e, d),
        "shash_K_scalar": np.array([K(float(p), 0.25) for p in inp["shash_p"]]),
        # P and K use dask with 2-D chunks, so they need 2-D input.
        "shash_K_array": np.asarray(K(inp["shash_p"][None, :], 0.25)),
        "shash_P": np.asarray(P(inp["shash_q"])),
    }
    m1, m2 = m1m2(inp["shash_eps2d"], inp["shash_delta2d"])
    out["shash_m1"], out["shash_m2"] = np.asarray(m1), np.asarray(m2)
    scalar = [
        m1m2(float(a), float(b))
        for a, b in zip(
            inp["shash_eps2d"].ravel(), inp["shash_delta2d"].ravel(), strict=True
        )
    ]
    out["shash_m1m2_scalar"] = np.array(scalar)
    return out


def _placeholder_priors(names: list[str]) -> list[Any]:
    """Build placeholder priors. The numpy maps ignore them.

    Parameters
    ----------
    names : list[str]
        Prior names.

    Returns
    -------
    list[Any]
        One prior per name.
    """
    return [make_prior(n) for n in names]


def likelihood_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate the numpy forward/backward/yhat maps of every likelihood.

    The parameter arrays have the (observations, sample) shape that HBR passes.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``lik_*`` input arrays and ``lik_zinb_seed``.

    Returns
    -------
    dict[str, np.ndarray]
        ``lik_<name>_forward``, ``_backward`` and (where defined) ``_yhat``.
    """
    mu, sigma, eps, delta = (
        inp["lik_mu"],
        inp["lik_sigma"],
        inp["lik_eps"],
        inp["lik_delta"],
    )
    Y, Z = inp["lik_Y"], inp["lik_Z"]
    out: dict[str, np.ndarray] = {}

    normal = NormalLikelihood(*_placeholder_priors(["mu", "sigma"]))
    out["lik_Normal_forward"] = normal.forward(mu, sigma, Y=Y)
    out["lik_Normal_backward"] = normal.backward(mu, sigma, Z=Z)
    out["lik_Normal_yhat"] = normal.yhat(mu, sigma)

    shashb = SHASHbLikelihood(*_placeholder_priors(["mu", "sigma", "epsilon", "delta"]))
    out["lik_SHASHb_forward"] = np.asarray(shashb.forward(mu, sigma, eps, delta, Y=Y))
    out["lik_SHASHb_backward"] = np.asarray(shashb.backward(mu, sigma, eps, delta, Z=Z))
    out["lik_SHASHb_yhat"] = shashb.yhat(mu, sigma, eps, delta)

    # SHASHo and SHASHo2 are abstract classes (missing methods), so call the
    # numpy maps unbound; they do not use self.
    out["lik_SHASHo_forward"] = SHASHoLikelihood.forward(
        None, mu, sigma, eps, delta, Y=Y
    )
    out["lik_SHASHo_backward"] = SHASHoLikelihood.backward(
        None, mu, sigma, eps, delta, Z=Z
    )
    out["lik_SHASHo2_forward"] = SHASHo2Likelihood.forward(
        None, mu, sigma, eps, delta, Y=Y
    )
    out["lik_SHASHo2_backward"] = SHASHo2Likelihood.backward(
        None, mu, sigma, eps, delta, Z=Z
    )

    beta = BetaLikelihood(*_placeholder_priors(["alpha", "beta"]))
    a, b = inp["lik_beta_alpha"], inp["lik_beta_beta"]
    out["lik_beta_forward"] = beta.forward(a, b, Y=inp["lik_Y_unit"])
    out["lik_beta_backward"] = beta.backward(a, b, Z=Z)
    out["lik_beta_yhat"] = beta.yhat(a, b)

    zinb = ZeroInflatedNegativeBinomialLikelihood(
        *_placeholder_priors(["mu", "alpha", "psi"])
    )
    zm, za, zp = inp["lik_zinb_mu"], inp["lik_zinb_alpha"], inp["lik_zinb_psi"]
    # ZINB forward draws random numbers; an explicit seeded rng makes it pure.
    rng = np.random.default_rng(int(inp["lik_zinb_seed"]))
    out["lik_ZINB_forward"] = zinb.forward(zm, za, zp, Y=inp["lik_Y_count"], rng=rng)
    out["lik_ZINB_backward"] = zinb.backward(zm, za, zp, Z=Z)
    out["lik_ZINB_yhat"] = zinb.yhat(zm, za, zp)
    return out


def golden_warps() -> dict[str, WarpBase]:
    """Return the warps that have golden values.

    Returns
    -------
    dict[str, WarpBase]
        Name to warp object.
    """
    return {
        "Log": WarpLog(),
        "Affine": WarpAffine(),
        "BoxCox": WarpBoxCox(),
        "SinhArcsinh": WarpSinhArcsinh(),
        "Compose": WarpCompose([WarpLog(), WarpAffine(), WarpSinhArcsinh()]),
    }


def warp_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate f, invf and df of each warp at fixed parameters.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``warp_x`` (positive values) and ``warp_param_<name>``.

    Returns
    -------
    dict[str, np.ndarray]
        ``warp_<name>_f``, ``_invf`` and ``_df``.
    """
    x = inp["warp_x"]
    out: dict[str, np.ndarray] = {}
    for name, warp in golden_warps().items():
        param = inp[f"warp_param_{name}"]
        y = warp.f(x, param)
        out[f"warp_{name}_f"] = y
        out[f"warp_{name}_invf"] = warp.invf(y, param)
        out[f"warp_{name}_df"] = warp.df(x, param)
    return out


# --------------------------------------------------------------------------- #
# Scalers and basis functions
# --------------------------------------------------------------------------- #

SCALER_CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "standardize": ("standardize", {}),
    "minmax": ("minmax", {}),
    "minmax_clip": ("minmax", {"adjust_outliers": True}),
    "robminmax": ("robminmax", {}),
    "robminmax_clip": ("robminmax", {"adjust_outliers": True, "tail": 0.1}),
    "id": ("id", {}),
}


def scaler_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit each scaler on training data and transform held-out data.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``scaler_train`` (n, 3) and ``scaler_test`` (m, 3).

    Returns
    -------
    dict[str, np.ndarray]
        ``scaler_<case>_transform``, ``_inverse`` and ``_transform_index``.
    """
    out: dict[str, np.ndarray] = {}
    index = np.array([0, 2])
    for case, (kind, kwargs) in SCALER_CASES.items():
        scaler = Scaler.from_string(kind, **kwargs)
        scaler.fit(inp["scaler_train"])
        transformed = scaler.transform(inp["scaler_test"])
        out[f"scaler_{case}_transform"] = transformed
        out[f"scaler_{case}_inverse"] = scaler.inverse_transform(transformed)
        if kind != "id":
            out[f"scaler_{case}_transform_index"] = scaler.transform(
                inp["scaler_test"][:, index], index=index
            )
    return out


def golden_basis_functions() -> dict[str, Any]:
    """Return the basis functions that have golden values.

    Returns
    -------
    dict[str, Any]
        Name to unfitted basis function.
    """
    return {
        "bspline": BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        "bspline_quantile": BsplineBasisFunction(
            basis_column=0, nknots=6, degree=2, knot_method="quantile"
        ),
        "bspline_linear": BsplineBasisFunction(
            basis_column=1, nknots=4, degree=3, include_linear=True
        ),
        "poly": PolynomialBasisFunction(basis_column=0, degree=3),
        "linear": LinearBasisFunction(basis_column=0),
    }


def basis_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit each basis function on training data and transform held-out data.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``basis_train`` and ``basis_test``, shape (n, 2).

    Returns
    -------
    dict[str, np.ndarray]
        ``basis_<name>``.
    """
    out: dict[str, np.ndarray] = {}
    for name, bf in golden_basis_functions().items():
        bf.fit(inp["basis_train"])
        out[f"basis_{name}"] = bf.transform(inp["basis_test"])
    return out


# --------------------------------------------------------------------------- #
# NormData scaling, Evaluator, batch-effect helpers
# --------------------------------------------------------------------------- #


def prediction_normdata(inp: dict[str, np.ndarray]) -> NormData:
    """Build a NormData with Y and every prediction field filled in.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``nd_X``, ``nd_be``, ``nd_Y`` (n, 2), ``nd_Yhat``, ``nd_Z``,
        ``nd_logp``, ``nd_baseline_logp`` (n, 2), ``nd_centiles`` (c, n, 2)
        and ``nd_centile_levels`` (c,).

    Returns
    -------
    NormData
        Data with two response variables.
    """
    data = to_normdata("golden_eval", inp["nd_X"], inp["nd_be"], inp["nd_Y"])
    obs_rv = ("observations", "response_vars")
    for var in ("Yhat", "Z", "logp", "baseline_logp"):
        data[var] = xr.DataArray(inp[f"nd_{var}"].copy(), dims=obs_rv)
    data["centiles"] = xr.DataArray(
        inp["nd_centiles"].copy(),
        dims=("centile", *obs_rv),
        coords={"centile": inp["nd_centile_levels"]},
    )
    return data


def normdata_scaling_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scale a NormData forward and back with fitted scalers.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        See ``prediction_normdata``.

    Returns
    -------
    dict[str, np.ndarray]
        ``nd_fwd_<var>`` and ``nd_bwd_<var>`` for X, Y, Yhat and centiles.
    """
    data = prediction_normdata(inp)
    inscalers: dict[str, Scaler] = {}
    for i, cov in enumerate(COVARIATES):
        inscalers[cov] = Scaler.from_string("standardize")
        inscalers[cov].fit(inp["nd_X"][:, i])
    outscalers: dict[str, Scaler] = {}
    for i, rv in enumerate(data.response_vars.values):
        outscalers[rv] = Scaler.from_string("minmax" if i else "standardize")
        outscalers[rv].fit(inp["nd_Y"][:, i])
    out: dict[str, np.ndarray] = {}
    data.scale_forward(inscalers, outscalers)
    for var in ("X", "Y", "Yhat", "centiles"):
        out[f"nd_fwd_{var}"] = data[var].values.copy()
    data.scale_backward(inscalers, outscalers)
    for var in ("X", "Y", "Yhat", "centiles"):
        out[f"nd_bwd_{var}"] = data[var].values.copy()
    return out


def evaluator_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run every Evaluator metric on a fixed prediction dataset.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        See ``prediction_normdata``.

    Returns
    -------
    dict[str, np.ndarray]
        ``eval_statistics`` (n_response_vars, n_statistics), ``eval_names``
        and ``eval_bic`` (BIC is not part of ``evaluate``).
    """
    data = prediction_normdata(inp)
    Evaluator().evaluate(data)
    bic = [
        Evaluator()._evaluate_bic(data.sel(response_vars=rv))
        for rv in data.response_vars.values
    ]
    return {
        "eval_statistics": data["statistics"].values.copy(),
        "eval_names": np.asarray(data["statistics"].statistic.values, dtype=str),
        "eval_bic": np.array(bic),
    }


def batch_combination_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """List batch-effect combinations and their masks.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``ibc_be`` (n, 2) strings; level lists ``ibc_levels_site`` and
        ``ibc_levels_sex`` (they include a level that never occurs).

    Returns
    -------
    dict[str, np.ndarray]
        ``ibc_combos`` (k, 2) strings and ``ibc_masks`` (k, n) booleans.
    """
    unique = {"site": list(inp["ibc_levels_site"]), "sex": list(inp["ibc_levels_sex"])}
    combos, masks = [], []
    for combo, mask in iter_batch_combinations(
        inp["ibc_be"], unique, BATCH_EFFECT_DIMS
    ):
        combos.append([combo[d] for d in BATCH_EFFECT_DIMS])
        masks.append(mask)
    return {"ibc_combos": np.array(combos, dtype=str), "ibc_masks": np.array(masks)}


def extract_and_reshape_outputs(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run ``HBR.extract_and_reshape`` on both of its input shapes.

    Parameters
    ----------
    inp : dict[str, np.ndarray]
        ``ear_in_global`` (n_samples,) and ``ear_in_per_subject``
        (n_observations, n_samples).

    Returns
    -------
    dict[str, np.ndarray]
        ``ear_global`` (repeated per observation) and ``ear_per_subject``.
    """
    n_obs = inp["ear_in_per_subject"].shape[0]
    post_pred = xr.Dataset(
        {
            "g": xr.DataArray(inp["ear_in_global"], dims=("sample",)),
            "p": xr.DataArray(
                inp["ear_in_per_subject"], dims=("observations", "sample")
            ),
        }
    )
    hbr = HBR("golden")
    return {
        "ear_global": hbr.extract_and_reshape(post_pred, n_obs, "g").values,
        "ear_per_subject": hbr.extract_and_reshape(post_pred, n_obs, "p").values,
    }


def sampling_outputs(model: NormativeModel, seed: int, n: int) -> dict[str, np.ndarray]:
    """Sample batch effects and covariates from a fitted model.

    Both functions use NumPy's legacy global random state, so this seeds it
    and restores the old state afterwards.

    Parameters
    ----------
    model : NormativeModel
        A fitted model.
    seed : int
        Seed for ``np.random.seed``.
    n : int
        Number of samples.

    Returns
    -------
    dict[str, np.ndarray]
        ``sample_bes`` (strings), ``sample_X`` and ``sample_X_per_be``.
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        bes = model.sample_batch_effects(n)
        X = model.sample_covariates(bes)
        X_per_be = model.sample_covariates(bes, covariate_range_per_batch_effect=True)
    finally:
        np.random.set_state(state)
    return {
        "sample_bes": np.asarray(bes.values, dtype=str),
        "sample_X": X.values.copy(),
        "sample_X_per_be": X_per_be.values.copy(),
    }
