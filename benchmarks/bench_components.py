"""Micro-benchmarks of hot-spot functions, each timed in isolation.

Groups (``--groups``; default all):

shash
    ``shash.K``, ``P``, ``m1m2``, ``S``, ``S_inv`` on (N, S) arrays, where S
    is the number of posterior samples. ``K``/``P``/``m1m2`` return lazy dask
    arrays; the timed call converts the result with ``np.asarray``, so the
    time includes the real computation.
likelihood
    numpy ``forward`` (Y to Z) and ``backward`` (Z to Y) of the Normal,
    SHASHb, SHASHo, SHASHo2, Beta and ZINB likelihoods on (N, S) parameter
    arrays, the same shapes HBR uses. ZINB forward gets a seeded ``rng``.
blr
    BLR internals at the fitted hyperparameters: ``post``, ``loglik``,
    ``penalized_loglik``, ``dloglik`` (constant-noise model only),
    ``ys_s2``, ``forward``, ``backward``, ``elemwise_logp``,
    ``compute_yhat`` and ``BLR.fit``. ``post``/``loglik`` cache on the
    hyperparameters; the cache is cleared before every call (untimed).
    ``BLR.fit`` is timed once, without warm-up: with 20 levels x 2 batch
    dimensions a heteroskedastic fit has ~50 hyperparameters and takes
    minutes at N=5000.
basis
    B-spline basis ``fit`` and ``transform``.
scaler
    standardize / minmax / robminmax ``fit``, ``transform``,
    ``inverse_transform``.
warp
    ``f``, ``invf``, ``df`` of WarpBoxCox, WarpSinhArcsinh, WarpAffine,
    WarpLog and WarpCompose.
normdata
    ``NormData.scale_forward`` / ``scale_backward`` on predicted data
    (X, Y, Yhat, centiles).
evaluator
    ``Evaluator.evaluate`` with all metrics, each metric alone, and BIC
    (``_evaluate_bic``; BIC is not in ``evaluate``'s metric list).
data_utils
    ``iter_batch_combinations`` over all batch-level combinations.
sampling
    ``NormativeModel.sample_covariates`` (uses global ``np.random``; it is
    seeded before every call so the work is the same).

Run from the repository root::

    python -m benchmarks.bench_components --size smoke
"""

from __future__ import annotations

import argparse
import copy
import tempfile
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from benchmarks.bench_blr import make_blr, new_model, predict_outputs
from benchmarks.data import make_synthetic
from benchmarks.timing import Timing, print_table, time_call, write_results
from pcntoolkit import BsplineBasisFunction, NormativeModel, NormData
from pcntoolkit.math_functions import shash
from pcntoolkit.math_functions.likelihood import (
    BetaLikelihood,
    NormalLikelihood,
    SHASHbLikelihood,
    SHASHo2Likelihood,
    SHASHoLikelihood,
    ZeroInflatedNegativeBinomialLikelihood,
)
from pcntoolkit.math_functions.prior import make_prior
from pcntoolkit.math_functions.scaler import Scaler
from pcntoolkit.math_functions.warp import (
    WarpAffine,
    WarpBoxCox,
    WarpCompose,
    WarpLog,
    WarpSinhArcsinh,
)
from pcntoolkit.util.data_utils import iter_batch_combinations
from pcntoolkit.util.evaluator import Evaluator
from pcntoolkit.util.output import Output

GROUPS = (
    "shash",
    "likelihood",
    "blr",
    "basis",
    "scaler",
    "warp",
    "normdata",
    "evaluator",
    "data_utils",
    "sampling",
)

# n: observations; s: posterior samples for the (N, S) arrays; r: response
# vars for NormData/Evaluator; levels: batch levels per dim (2 dims);
# n_sampling: observations for sample_covariates (a Python loop per row).
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "n": 1000,
        "s": 200,
        "r": 2,
        "levels": 2,
        "n_sampling": 200,
        "repeats": 3,
        "warmup": 1,
    },
    "small": {
        "n": 5000,
        "s": 1000,
        "r": 10,
        "levels": 20,
        "n_sampling": 1000,
        "repeats": 5,
        "warmup": 1,
    },
    "medium": {
        "n": 10000,
        "s": 2000,
        "r": 10,
        "levels": 20,
        "n_sampling": 2000,
        "repeats": 5,
        "warmup": 1,
    },
    "large": {
        "n": 50000,
        "s": 1000,
        "r": 100,
        "levels": 20,
        "n_sampling": 5000,
        "repeats": 3,
        "warmup": 1,
    },
}

EVAL_METRICS = (
    "Rho",
    "R2",
    "RMSE",
    "SMSE",
    "EXPV",
    "MSLL",
    "MLL",
    "ShapiroW",
    "MACE",
    "MAPE",
    "Skewness",
    "Kurtosis",
)


def _force(x: Any) -> np.ndarray:
    """Materialise a numpy, dask or xarray result.

    Parameters
    ----------
    x : Any
        Result of the timed function.

    Returns
    -------
    np.ndarray
        Computed array.
    """
    if isinstance(x, xr.DataArray):
        x = x.data
    return np.asarray(x)


class Bench:
    """Collects timings of one group.

    Parameters
    ----------
    group : str
        Group name, stored in each case.
    repeats : int
        Timed calls per operation.
    warmup : int
        Warm-up calls per operation.

    Attributes
    ----------
    cases : list[dict[str, Any]]
        One case per timed operation.
    """

    def __init__(self, group: str, repeats: int, warmup: int) -> None:
        self.group = group
        self.repeats = repeats
        self.warmup = warmup
        self.cases: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        fn: Callable[[], Any],
        setup: Callable[[], Any] | None = None,
        once: bool = False,
        **shape: Any,
    ) -> Timing:
        """Time ``fn`` and store it as a case.

        Parameters
        ----------
        name : str
            Operation name.
        fn : Callable[[], Any]
            Function to time; must force lazy results itself.
        setup : Callable[[], Any] | None
            Untimed function run before each call.
        once : bool
            If True, time one call without warm-up (for slow operations).
        **shape : Any
            Size information stored in the case.

        Returns
        -------
        Timing
            The timing.
        """
        repeats, warmup = (1, 0) if once else (self.repeats, self.warmup)
        t = time_call(name, fn, repeats, warmup, setup)
        self.cases.append({"case": {"group": self.group, **shape}, "timings": [t]})
        return t


def bench_shash(b: Bench, n: int, s: int, rng: np.random.Generator) -> None:
    """Time the SHASH helper functions.

    Parameters
    ----------
    b : Bench
        Collector.
    n : int
        Observations.
    s : int
        Posterior samples.
    rng : np.random.Generator
        Random source.
    """
    eps = rng.normal(0.0, 0.3, (n, s))
    delta = rng.uniform(0.6, 1.5, (n, s))
    x = rng.normal(0.0, 1.0, (n, s))
    shape = {"n": n, "s": s}
    b.add(
        "shash.K",
        lambda: _force(shash.K((1.0 / delta + 1) / 2, 0.25, chunks=(1000, 1000))),
        **shape,
    )
    b.add("shash.P", lambda: _force(shash.P(1.0 / delta)), **shape)
    b.add("shash.m1m2", lambda: [_force(v) for v in shash.m1m2(eps, delta)], **shape)
    b.add("shash.S", lambda: shash.S(x, eps, delta), **shape)
    b.add("shash.S_inv", lambda: shash.S_inv(x, eps, delta), **shape)


class _Unbound:
    """Call ``forward``/``backward`` of a likelihood class without an instance.

    SHASHoLikelihood and SHASHo2Likelihood are abstract (they miss
    ``compile_params``, ``transfer``, ``yhat``, ``_update_data``), so they
    cannot be instantiated. Their ``forward``/``backward`` do not use
    ``self``, so we call them as plain functions.

    Parameters
    ----------
    cls : type
        Likelihood class.
    """

    def __init__(self, cls: type) -> None:
        self.cls = cls

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``cls.forward`` with ``self=None``.

        Parameters
        ----------
        *args : Any
            Parameter arrays.
        **kwargs : Any
            ``Y``.

        Returns
        -------
        Any
            Z-scores.
        """
        return self.cls.forward(None, *args, **kwargs)

    def backward(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``cls.backward`` with ``self=None``.

        Parameters
        ----------
        *args : Any
            Parameter arrays.
        **kwargs : Any
            ``Z``.

        Returns
        -------
        Any
            Y values.
        """
        return self.cls.backward(None, *args, **kwargs)


def bench_likelihood(b: Bench, n: int, s: int, rng: np.random.Generator) -> None:
    """Time numpy forward/backward of each likelihood.

    Parameters
    ----------
    b : Bench
        Collector.
    n : int
        Observations.
    s : int
        Posterior samples.
    rng : np.random.Generator
        Random source.
    """
    p = make_prior
    mu = rng.normal(0.0, 1.0, (n, s))
    sigma = rng.uniform(0.5, 2.0, (n, s))
    eps = rng.normal(0.0, 0.3, (n, s))
    delta = rng.uniform(0.6, 1.5, (n, s))
    y = rng.normal(0.0, 1.0, (n, 1))
    z = rng.normal(0.0, 1.0, (n, 1))
    a_beta = rng.uniform(1.0, 5.0, (n, s))
    b_beta = rng.uniform(1.0, 5.0, (n, s))
    y_beta = rng.uniform(0.01, 0.99, (n, 1))
    mu_zinb = rng.uniform(1.0, 10.0, (n, s))
    alpha_zinb = rng.uniform(0.5, 5.0, (n, s))
    psi_zinb = rng.uniform(0.5, 0.95, (n, s))
    y_zinb = rng.poisson(4.0, (n, 1)).astype(float)
    cases = [
        ("Normal", NormalLikelihood(p(), p()), (mu, sigma), y),
        ("SHASHb", SHASHbLikelihood(p(), p(), p(), p()), (mu, sigma, eps, delta), y),
        ("SHASHo", _Unbound(SHASHoLikelihood), (mu, sigma, eps, delta), y),
        ("SHASHo2", _Unbound(SHASHo2Likelihood), (mu, sigma, eps, delta), y),
        ("Beta", BetaLikelihood(p(), p()), (a_beta, b_beta), y_beta),
        (
            "ZINB",
            ZeroInflatedNegativeBinomialLikelihood(p(), p(), p()),
            (mu_zinb, alpha_zinb, psi_zinb),
            y_zinb,
        ),
    ]
    shape = {"n": n, "s": s}
    for name, lik, params, yy in cases:
        extra = {"rng": np.random.default_rng(0)} if name == "ZINB" else {}
        b.add(
            f"{name}.forward",
            lambda lik=lik, params=params, yy=yy, extra=extra: _force(
                lik.forward(*params, Y=yy, **extra)
            ),
            **shape,
        )
        b.add(
            f"{name}.backward",
            lambda lik=lik, params=params: _force(lik.backward(*params, Z=z)),
            **shape,
        )


def _fitted_blr(
    config: str, n: int, levels: int, seed: int, scratch: str
) -> tuple[NormativeModel, NormData, NormData]:
    """Fit a one-response BLR normative model on synthetic data.

    Parameters
    ----------
    config : str
        BLR configuration name (see ``bench_blr.make_blr``).
    n : int
        Observations.
    levels : int
        Batch levels per dimension (2 dimensions).
    seed : int
        Data seed.
    scratch : str
        Folder the model may create sub-folders in.

    Returns
    -------
    tuple[NormativeModel, NormData, NormData]
        Fitted model, scaled training data (one response var) and unscaled
        test data.
    """
    train = make_synthetic(n, 1, levels, 2, seed, 0, "train")
    test = make_synthetic(n, 1, levels, 2, seed, 1, "test")
    nm = new_model(config, scratch)
    nm.fit(copy.deepcopy(train))
    nm.preprocess(train)
    return nm, train, test


def bench_blr(b: Bench, n: int, levels: int, seed: int, scratch: str) -> None:
    """Time BLR internals at the fitted hyperparameters.

    Parameters
    ----------
    b : Bench
        Collector.
    n : int
        Observations.
    levels : int
        Batch levels per dimension.
    seed : int
        Data seed.
    scratch : str
        Scratch folder.
    """
    for config in ("plain", "hetero", "warp"):
        _bench_blr_config(b, config, n, levels, seed, scratch)


def _bench_blr_config(
    b: Bench, config: str, n: int, levels: int, seed: int, scratch: str
) -> None:
    """Time BLR internals for one configuration.

    Parameters
    ----------
    b : Bench
        Collector.
    config : str
        BLR configuration name.
    n : int
        Observations.
    levels : int
        Batch levels per dimension.
    seed : int
        Data seed.
    scratch : str
        Scratch folder.
    """
    nm, train, _ = _fitted_blr(config, n, levels, seed, scratch)
    rv = nm.response_vars[0]
    reg = nm[rv]
    X, be, be_maps, Y, _ = nm.extract_data(train.sel({"response_vars": rv}))
    Phi, Phi_var = reg.Phi_Phi_var(X.values, be.values)
    y = Y.values
    hyp = reg.hyp.copy()
    shape = {
        "n": n,
        "levels": levels,
        "config": config,
        "n_hyp": int(hyp.size),
        "d": int(Phi.shape[1]),
    }

    def clear() -> None:
        # post/loglik skip work when hyp is unchanged and Sigma_a exists.
        reg.__dict__.pop("Sigma_a", None)

    b.add("BLR.post", lambda: reg.post(hyp, Phi, y, Phi_var), clear, **shape)
    b.add(
        "BLR.loglik (incl. post)",
        lambda: reg.loglik(hyp, Phi, y, Phi_var),
        clear,
        **shape,
    )
    b.add(
        "BLR.penalized_loglik (incl. post)",
        lambda: reg.penalized_loglik(hyp, Phi, y, Phi_var, 0.1, "l2"),
        clear,
        **shape,
    )
    # dloglik builds dense N x N matrices (np.diag of length N), so skip it for large N.
    if config == "plain" and n <= 10000:
        b.add(
            "BLR.dloglik (incl. post)",
            lambda: reg.dloglik(hyp, Phi, y, Phi_var),
            clear,
            **shape,
        )
    reg.loglik(hyp, Phi, y, Phi_var)
    b.add("BLR.ys_s2", lambda: reg.ys_s2(X.values, be.values), **shape)
    b.add("BLR.forward", lambda: reg.forward(X, be, Y), **shape)
    zz = xr.DataArray(np.full(X.shape[0], 1.0), dims=("observations",))
    b.add("BLR.backward", lambda: reg.backward(X, be, zz), **shape)
    b.add("BLR.elemwise_logp", lambda: reg.elemwise_logp(X, be, Y), **shape)
    rdata = train.sel({"response_vars": rv})
    b.add(
        "BLR.compute_yhat (200 backward)",
        lambda: reg.compute_yhat(rdata, rv, X, be),
        **shape,
    )
    state: dict[str, Any] = {}

    def fresh() -> None:
        state["m"] = make_blr(config)

    # A heteroskedastic fit with many batch levels takes minutes; time it once.
    b.add(
        "BLR.fit", lambda: state["m"].fit(X, be, be_maps, Y), fresh, once=True, **shape
    )


def bench_basis(b: Bench, n: int, rng: np.random.Generator) -> None:
    """Time B-spline basis fit and transform.

    Parameters
    ----------
    b : Bench
        Collector.
    n : int
        Observations.
    rng : np.random.Generator
        Random source.
    """
    x = rng.uniform(-2.0, 2.0, (n, 1))
    bf = BsplineBasisFunction(degree=3, nknots=5)
    bf.fit(x)
    b.add("Bspline.fit", lambda: BsplineBasisFunction(degree=3, nknots=5).fit(x), n=n)
    b.add("Bspline.transform", lambda: bf.transform(x), n=n)


def bench_scaler(b: Bench, n: int, rng: np.random.Generator) -> None:
    """Time scaler fit/transform/inverse_transform.

    Parameters
    ----------
    b : Bench
        Collector.
    n : int
        Observations.
    rng : np.random.Generator
        Random source.
    """
    x = rng.normal(3.0, 2.0, n)
    for kind in ("standardize", "minmax", "robminmax"):
        sc = Scaler.from_string(kind)
        sc.fit(x)
        xt = sc.transform(x)
        b.add(f"{kind}.fit", lambda kind=kind: Scaler.from_string(kind).fit(x), n=n)
        b.add(f"{kind}.transform", lambda sc=sc: sc.transform(x), n=n)
        b.add(
            f"{kind}.inverse_transform",
            lambda sc=sc, xt=xt: sc.inverse_transform(xt),
            n=n,
        )


def bench_warp(b: Bench, n: int, rng: np.random.Generator) -> None:
    """Time warp f/invf/df.

    Parameters
    ----------
    b : Bench
        Collector.
    n : int
        Observations.
    rng : np.random.Generator
        Random source.
    """
    x = rng.uniform(0.5, 5.0, n)
    warps = [
        ("WarpBoxCox", WarpBoxCox(), np.array([np.log(0.5)])),
        ("WarpSinhArcsinh", WarpSinhArcsinh(), np.array([0.2, 0.1])),
        ("WarpAffine", WarpAffine(), np.array([0.5, 0.2])),
        ("WarpLog", WarpLog(), np.array([])),
        (
            "WarpCompose",
            WarpCompose([WarpBoxCox(), WarpAffine(), WarpSinhArcsinh()]),
            np.array([np.log(0.5), 0.5, 0.2, 0.2, 0.1]),
        ),
    ]
    for name, w, prm in warps:
        fx = w.f(x, prm)
        b.add(f"{name}.f", lambda w=w, prm=prm: w.f(x, prm), n=n)
        b.add(f"{name}.invf", lambda w=w, prm=prm, fx=fx: w.invf(fx, prm), n=n)
        b.add(f"{name}.df", lambda w=w, prm=prm: w.df(x, prm), n=n)


def _predicted(
    n: int, r: int, levels: int, seed: int, scratch: str
) -> tuple[NormativeModel, NormData]:
    """Fit a BLR model on synthetic data and predict a test set.

    Parameters
    ----------
    n : int
        Observations.
    r : int
        Response variables.
    levels : int
        Batch levels per dimension (2 dimensions).
    seed : int
        Data seed.
    scratch : str
        Folder the model may create sub-folders in.

    Returns
    -------
    tuple[NormativeModel, NormData]
        Fitted model and predicted (unscaled) test data.
    """
    train = make_synthetic(n, r, levels, 2, seed, 0, "train")
    test = make_synthetic(n, r, levels, 2, seed, 1, "test")
    nm = new_model("plain", scratch)
    nm.fit(copy.deepcopy(train))
    return nm, predict_outputs(nm, test)


def bench_normdata(
    b: Bench, nm: NormativeModel, data: NormData, n: int, r: int
) -> None:
    """Time NormData.scale_forward and scale_backward.

    Parameters
    ----------
    b : Bench
        Collector.
    nm : NormativeModel
        Fitted model (provides the scalers).
    data : NormData
        Predicted, unscaled data.
    n, r : int
        Sizes for the case label.
    """
    d = copy.deepcopy(data)
    ins, outs = nm.inscalers, nm.outscalers
    b.add(
        "NormData.scale_forward",
        lambda: d.scale_forward(ins, outs),
        lambda: d.scale_backward(ins, outs),
        n=n,
        r=r,
    )
    b.add(
        "NormData.scale_backward",
        lambda: d.scale_backward(ins, outs),
        lambda: d.scale_forward(ins, outs),
        n=n,
        r=r,
    )


def bench_evaluator(b: Bench, data: NormData, n: int, r: int) -> None:
    """Time Evaluator.evaluate, each metric, and BIC.

    Parameters
    ----------
    b : Bench
        Collector.
    data : NormData
        Predicted data.
    n, r : int
        Sizes for the case label.
    """
    d = copy.deepcopy(data)
    ev = Evaluator()
    b.add("Evaluator.evaluate (all)", lambda: ev.evaluate(d), n=n, r=r)
    for m in EVAL_METRICS:
        b.add(f"Evaluator.{m}", lambda m=m: ev.evaluate(d, statistics=[m]), n=n, r=r)
    b.add(
        "Evaluator._evaluate_bic",
        lambda: [
            ev._evaluate_bic(d.sel(response_vars=rv)) for rv in d.response_var_list
        ],
        n=n,
        r=r,
    )


def bench_data_utils(b: Bench, n: int, levels: int, rng: np.random.Generator) -> None:
    """Time iter_batch_combinations over two batch-effect dimensions.

    Parameters
    ----------
    b : Bench
        Collector.
    n : int
        Observations.
    levels : int
        Levels per dimension.
    rng : np.random.Generator
        Random source.
    """
    values = rng.integers(0, levels, (n, 2))
    unique = {"a": list(range(levels)), "b": list(range(levels))}
    b.add(
        "iter_batch_combinations",
        lambda: list(iter_batch_combinations(values, unique, ["a", "b"])),
        n=n,
        levels=levels,
    )


def bench_sampling(b: Bench, nm: NormativeModel, n: int) -> None:
    """Time NormativeModel.sample_covariates (seeded global RNG).

    Parameters
    ----------
    b : Bench
        Collector.
    nm : NormativeModel
        Fitted model.
    n : int
        Number of rows to sample.
    """
    np.random.seed(0)
    bes = nm.sample_batch_effects(n)
    for per_be in (False, True):
        b.add(
            f"sample_covariates(per_batch_effect={per_be})",
            lambda per_be=per_be: nm.sample_covariates(bes, per_be),
            lambda: np.random.seed(0),
            n=n,
        )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the groups, write JSON and print a table.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; default ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeats", type=int, help="timed calls per operation (default: preset)"
    )
    parser.add_argument(
        "--warmup", type=int, help="warm-up calls per operation (default: preset)"
    )
    parser.add_argument(
        "--scratch", help="folder for temporary model files (default: system temp)"
    )
    parser.add_argument(
        "--out", help="result JSON path (default: benchmarks/results/...)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="show pcntoolkit messages and warnings"
    )
    args = parser.parse_args(argv)
    pre = PRESETS[args.size]
    args.repeats = pre["repeats"] if args.repeats is None else args.repeats
    args.warmup = pre["warmup"] if args.warmup is None else args.warmup
    if not args.verbose:
        Output.set_show_messages(False)
        Output.set_show_warnings(False)
        warnings.filterwarnings("ignore")

    t_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    cases: list[dict[str, Any]] = []
    predicted: tuple[NormativeModel, NormData] | None = None
    tmp = tempfile.TemporaryDirectory(prefix="pcnbench_comp_", dir=args.scratch)
    for group in args.groups:
        b = Bench(group, args.repeats, args.warmup)
        if group == "shash":
            bench_shash(b, pre["n"], pre["s"], rng)
        elif group == "likelihood":
            bench_likelihood(b, pre["n"], pre["s"], rng)
        elif group == "blr":
            bench_blr(b, pre["n"], pre["levels"], args.seed, tmp.name)
        elif group == "basis":
            bench_basis(b, pre["n"], rng)
        elif group == "scaler":
            bench_scaler(b, pre["n"], rng)
        elif group == "warp":
            bench_warp(b, pre["n"], rng)
        elif group == "data_utils":
            bench_data_utils(b, pre["n"], pre["levels"], rng)
        else:
            if predicted is None:
                predicted = _predicted(
                    pre["n"], pre["r"], pre["levels"], args.seed, tmp.name
                )
            nm, data = predicted
            if group == "normdata":
                bench_normdata(b, nm, data, pre["n"], pre["r"])
            elif group == "evaluator":
                bench_evaluator(b, data, pre["n"], pre["r"])
            elif group == "sampling":
                bench_sampling(b, nm, pre["n_sampling"])
        cases.extend(b.cases)
        print(f"done {group}", flush=True)
    tmp.cleanup()

    params = {**vars(args), "preset": pre, "wall_time_s": time.perf_counter() - t_start}
    path = write_results("bench_components", params, cases, args.out)
    print_table(cases)
    print(f"wrote {path}  (total {params['wall_time_s']:.1f} s)")


if __name__ == "__main__":
    main()
