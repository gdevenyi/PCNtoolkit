"""Benchmark HBR normative models: predict from a saved model, and fit.

Two stages (``--stages``; default both):

predict
    Fits a model once and saves it to ``benchmarks/models/<key>/``
    (git-ignored). Later runs reuse it, so two code versions predict from
    the same posterior draws. Then times, on a test set:
    ``load``, ``compute_zscores``, ``compute_centiles`` (``recompute=True``),
    ``compute_logp``, ``compute_yhat`` and ``predict`` (saving, plotting and
    evaluation off). Warm-up times are reported apart: the first call
    compiles PyTensor functions. Each case also checks that two loads of
    the saved model predict the same values (DETERMINISTIC class).
fit
    Times the regression fits only (``HBR.fit`` per response variable after
    ``preprocess``; no predict, no save). This includes building the PyMC
    model, sampler compilation (nutpie/numba) and sampling. Short runs by
    default; the ``medium`` and ``large`` presets add one run with the
    library default sampling (tune=500, draws=1500, chains=4) at N=1000,
    R=1.

Run from the repository root::

    python -m benchmarks.bench_hbr --size smoke
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

from benchmarks.compare import PREDICTION_VARS, compare_deterministic
from benchmarks.data import MODELS_DIR, make_dataset
from benchmarks.env import git_info
from benchmarks.timing import Timing, print_table, time_call, write_results
from pcntoolkit import (
    HBR,
    BsplineBasisFunction,
    NormativeModel,
    NormData,
    SHASHbLikelihood,
    make_prior,
)
from pcntoolkit.math_functions.likelihood import get_default_normal_likelihood
from pcntoolkit.util.output import Output

LIBRARY_DEFAULT_SAMPLING = {"tune": 500, "draws": 1500, "chains": 4}

# predict: sizes of saved models and the sampling used to fit them.
# fit: sizes and sampling of the timed fits; "default_fit" adds one fit with
# the library default sampling at N=1000, R=1, 2 levels.
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "predict": {
            "n": [200],
            "r": [1],
            "levels": [2],
            "sampling": {"tune": 50, "draws": 50, "chains": 2},
        },
        "fit": {
            "n": [200],
            "r": [1],
            "levels": [2],
            "sampling": {"tune": 50, "draws": 50, "chains": 2},
        },
        "default_fit": False,
        "repeats": 2,
        "warmup": 1,
    },
    "small": {
        "predict": {
            "n": [1000],
            "r": [1],
            "levels": [2, 20],
            "sampling": {"tune": 200, "draws": 200, "chains": 2},
        },
        "fit": {
            "n": [1000],
            "r": [1],
            "levels": [2, 20],
            "sampling": {"tune": 200, "draws": 200, "chains": 2},
        },
        "default_fit": False,
        "repeats": 3,
        "warmup": 1,
    },
    "medium": {
        "predict": {
            "n": [1000, 10000],
            "r": [1],
            "levels": [2, 20],
            "sampling": {"tune": 500, "draws": 1000, "chains": 2},
        },
        "fit": {
            "n": [1000, 10000],
            "r": [1],
            "levels": [2, 20],
            "sampling": {"tune": 200, "draws": 200, "chains": 2},
        },
        "default_fit": True,
        "repeats": 3,
        "warmup": 1,
    },
    "large": {
        "predict": {
            "n": [1000, 10000, 50000],
            "r": [1, 10],
            "levels": [2, 20],
            "sampling": LIBRARY_DEFAULT_SAMPLING,
        },
        "fit": {
            "n": [1000, 10000, 50000],
            "r": [1],
            "levels": [2, 20],
            "sampling": {"tune": 200, "draws": 200, "chains": 2},
        },
        "default_fit": True,
        "repeats": 3,
        "warmup": 1,
    },
}


def make_shashb_likelihood() -> SHASHbLikelihood:
    """SHASHb likelihood as in the HBR SHASH tutorial (fixed epsilon/delta).

    Returns
    -------
    SHASHbLikelihood
        Likelihood with a B-spline mean with a random intercept per batch
        level, a B-spline heteroskedastic SD, and scalar skew (epsilon) and
        tail (delta) parameters.
    """
    bspline = BsplineBasisFunction(basis_column=0, nknots=5, degree=3)
    mu = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
        intercept=make_prior(
            random=True,
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
            sigma=make_prior(
                dist_name="Normal",
                dist_params=(0.0, 1.0),
                mapping="softplus",
                mapping_params=(0.0, 3.0),
            ),
        ),
        basis_function=bspline,
    )
    sigma = make_prior(
        linear=True,
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        basis_function=copy.deepcopy(bspline),
        mapping="softplus",
        mapping_params=(0.0, 3.0),
    )
    epsilon = make_prior(dist_name="Normal", dist_params=(0.0, 1.0))
    delta = make_prior(
        dist_name="Normal",
        dist_params=(1.0, 1.0),
        mapping="softplus",
        mapping_params=(0.0, 3.0, 0.6),
    )
    return SHASHbLikelihood(mu, sigma, epsilon, delta)


def make_hbr(
    likelihood: str, tune: int, draws: int, chains: int, sampler: str = "nutpie"
) -> HBR:
    """Build an HBR template.

    Parameters
    ----------
    likelihood : str
        ``"normal"`` (library default Normal model) or ``"shashb"``.
    tune : int
        Tuning (warm-up) steps per chain; these draws are discarded.
    draws : int
        Kept draws per chain.
    chains : int
        Number of chains; also used as the number of cores.
    sampler : str
        NUTS implementation, e.g. ``"nutpie"`` or ``"pymc"``.

    Returns
    -------
    HBR
        Unfitted template model.

    Raises
    ------
    ValueError
        If ``likelihood`` is not known.
    """
    if likelihood == "normal":
        lik = get_default_normal_likelihood()
    elif likelihood == "shashb":
        lik = make_shashb_likelihood()
    else:
        raise ValueError(f"Unknown likelihood {likelihood!r}")
    return HBR(
        likelihood=lik,
        tune=tune,
        draws=draws,
        chains=chains,
        cores=chains,
        nuts_sampler=sampler,
        progressbar=False,
    )


def new_model(hbr: HBR, save_dir: str, savemodel: bool = False) -> NormativeModel:
    """Wrap an HBR template in a NormativeModel with plots/results off.

    Parameters
    ----------
    hbr : HBR
        Template regression model.
    save_dir : str
        Folder for the model.
    savemodel : bool
        Whether ``fit`` saves the model.

    Returns
    -------
    NormativeModel
        Unfitted model.
    """
    return NormativeModel(
        hbr,
        savemodel=savemodel,
        evaluate_model=False,
        saveresults=False,
        saveplots=False,
        save_dir=save_dir,
    )


def model_key(
    source: str,
    likelihood: str,
    n: int,
    r: int,
    levels: int,
    args: argparse.Namespace,
    sampling: dict[str, int],
) -> str:
    """Folder name of a saved model; same inputs give the same folder.

    Parameters
    ----------
    source : str
        Data source.
    likelihood : str
        Likelihood name.
    n, r, levels : int
        Data sizes.
    args : argparse.Namespace
        Command line (seed, batch dims, sampler).
    sampling : dict[str, int]
        tune, draws, chains.

    Returns
    -------
    str
        Folder name.
    """
    size = (
        f"n{n}_r{r}_l{levels}_bd{args.batch_dims}" if source == "synthetic" else f"r{r}"
    )
    s = sampling
    run = f"t{s['tune']}_d{s['draws']}_c{s['chains']}_{args.sampler}"
    return f"hbr_{likelihood}_{source}_{size}_seed{args.seed}_{run}"


def get_saved_model(
    key: str,
    train: NormData,
    likelihood: str,
    sampling: dict[str, int],
    args: argparse.Namespace,
) -> tuple[Path, bool]:
    """Return the folder of a saved fitted model; fit and save it if missing.

    Parameters
    ----------
    key : str
        Folder name under ``benchmarks/models``.
    train : NormData
        Training data.
    likelihood : str
        Likelihood name.
    sampling : dict[str, int]
        tune, draws, chains.
    args : argparse.Namespace
        Command line.

    Returns
    -------
    tuple[Path, bool]
        Model folder and whether it was fitted in this run.
    """
    path = Path(args.models_dir) / key
    if (path / "model" / "normative_model.json").exists() and not args.refit:
        return path, False
    if path.exists():
        shutil.rmtree(path)
    hbr = make_hbr(likelihood, sampler=args.sampler, **sampling)
    model = new_model(hbr, str(path), savemodel=True)
    model.fit(copy.deepcopy(train))
    (path / "bench_meta.json").write_text(
        json.dumps({"git": git_info(), "sampling": sampling}, indent=2)
    )
    return path, True


def predict_outputs(model: NormativeModel, data: NormData) -> NormData:
    """Run the predict steps on a copy of ``data`` and return it.

    Parameters
    ----------
    model : NormativeModel
        Fitted model.
    data : NormData
        Test data.

    Returns
    -------
    NormData
        Copy with Z, centiles, baseline_logp, logp and Yhat.
    """
    d = copy.deepcopy(data)
    model.compute_zscores(d)
    model.compute_centiles(d, recompute=True)
    model.compute_baseline_logp(d)
    model.compute_logp(d)
    model.compute_yhat(d)
    return d


def run_predict_case(
    source: str, likelihood: str, n: int, r: int, levels: int, args: argparse.Namespace
) -> dict[str, Any]:
    """Time the predict-side operations of one saved model.

    Parameters
    ----------
    source : str
        Data source.
    likelihood : str
        Likelihood name.
    n, r, levels : int
        Data sizes.
    args : argparse.Namespace
        Command line.

    Returns
    -------
    dict[str, Any]
        Case, timings and checks.
    """
    sampling = args.preset["predict"]["sampling"]
    train, test = make_dataset(source, n, r, levels, args.batch_dims, args.seed)
    key = model_key(source, likelihood, n, r, levels, args, sampling)
    t0 = time.perf_counter()
    path, fitted_now = get_saved_model(key, train, likelihood, sampling, args)
    case = {
        "stage": "predict",
        "data": source,
        "likelihood": likelihood,
        "n_test": int(test.X.shape[0]),
        "r": r,
        "levels": levels if source == "synthetic" else "fcon",
        "samples": sampling["draws"] * sampling["chains"],
    }
    rep, warm = args.repeats, args.warmup
    timings: list[Timing] = [
        time_call("load", lambda: NormativeModel.load(str(path)), rep, warm)
    ]
    loaded = NormativeModel.load(str(path))
    d = copy.deepcopy(test)
    timings.append(
        time_call("compute_zscores", lambda: loaded.compute_zscores(d), rep, warm)
    )
    timings.append(
        time_call(
            "compute_centiles",
            lambda: loaded.compute_centiles(d, recompute=True),
            rep,
            warm,
        )
    )
    timings.append(time_call("compute_logp", lambda: loaded.compute_logp(d), rep, warm))
    timings.append(time_call("compute_yhat", lambda: loaded.compute_yhat(d), rep, warm))
    timings.append(time_call("predict", lambda: loaded.predict(d), rep, warm))

    out_a = predict_outputs(NormativeModel.load(str(path)), test)
    out_b = predict_outputs(NormativeModel.load(str(path)), test)
    checks = [
        compare_deterministic(out_a[v], out_b[v], name=f"reload_{v}").__dict__
        for v in PREDICTION_VARS
        if v in out_b
    ]
    if args.save_outputs:
        outdir = Path(args.save_outputs)
        outdir.mkdir(parents=True, exist_ok=True)
        out_a.to_netcdf(str(outdir / f"{key}.nc"))
    return {
        "case": case,
        "timings": timings,
        "checks": checks,
        "model_dir": str(path),
        "model_fitted_this_run": fitted_now,
        "model_prepare_s": time.perf_counter() - t0 if fitted_now else None,
    }


def run_fit_case(
    source: str,
    likelihood: str,
    n: int,
    r: int,
    levels: int,
    sampling: dict[str, int],
    args: argparse.Namespace,
    scratch: Path,
) -> dict[str, Any]:
    """Time the regression fits of one case.

    Parameters
    ----------
    source : str
        Data source.
    likelihood : str
        Likelihood name.
    n, r, levels : int
        Data sizes.
    sampling : dict[str, int]
        tune, draws, chains.
    args : argparse.Namespace
        Command line.
    scratch : Path
        Scratch folder.

    Returns
    -------
    dict[str, Any]
        Case and timings.
    """
    train, _ = make_dataset(source, n, r, levels, args.batch_dims, args.seed)
    state: dict[str, Any] = {}

    def fresh() -> None:
        state["model"] = new_model(
            make_hbr(likelihood, sampler=args.sampler, **sampling), str(scratch)
        )
        state["data"] = copy.deepcopy(train)

    def fit_only() -> None:
        model, data = state["model"], state["data"]
        model.register_data_info(data)
        model.preprocess(data)
        for rv in model.response_vars:
            X, be, be_maps, Y, _ = model.extract_data(data.sel({"response_vars": rv}))
            model[rv].fit(X, be, be_maps, Y)

    case = {
        "stage": "fit",
        "data": source,
        "likelihood": likelihood,
        "n_train": int(train.X.shape[0]),
        "r": r,
        "levels": levels if source == "synthetic" else "fcon",
        **sampling,
    }
    timing = time_call("fit_only", fit_only, args.fit_repeats, args.fit_warmup, fresh)
    return {"case": case, "timings": [timing]}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the stages, write JSON and print a table.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; default ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--data", choices=("synthetic", "fcon"), default="synthetic")
    parser.add_argument(
        "--stages", nargs="+", choices=("predict", "fit"), default=["predict", "fit"]
    )
    parser.add_argument(
        "--likelihoods",
        nargs="+",
        choices=("normal", "shashb"),
        default=["normal", "shashb"],
    )
    parser.add_argument(
        "--sampler", default="nutpie", help="nuts_sampler passed to HBR"
    )
    parser.add_argument(
        "--batch-dims", type=int, default=1, help="batch-effect dimensions (synthetic)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the synthetic data (MCMC itself is not seeded)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="timed calls per predict operation (default: preset)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        help="warm-up calls per predict operation (default: preset)",
    )
    parser.add_argument(
        "--fit-repeats", type=int, default=1, help="timed fits per fit case"
    )
    parser.add_argument(
        "--fit-warmup", type=int, default=0, help="untimed fits before the timed ones"
    )
    parser.add_argument(
        "--models-dir", default=str(MODELS_DIR), help="where saved models live"
    )
    parser.add_argument(
        "--refit", action="store_true", help="refit saved models even if present"
    )
    parser.add_argument(
        "--scratch", help="folder for temporary files (default: system temp)"
    )
    parser.add_argument("--save-outputs", help="write test predictions (.nc) here")
    parser.add_argument(
        "--out", help="result JSON path (default: benchmarks/results/...)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show pcntoolkit/PyMC messages and warnings",
    )
    args = parser.parse_args(argv)
    args.preset = PRESETS[args.size]
    args.repeats = args.preset["repeats"] if args.repeats is None else args.repeats
    args.warmup = args.preset["warmup"] if args.warmup is None else args.warmup
    if not args.verbose:
        Output.set_show_messages(False)
        Output.set_show_warnings(False)
        warnings.filterwarnings("ignore")
        for name in ("pymc", "pytensor", "nutpie"):
            logging.getLogger(name).setLevel(logging.ERROR)

    t_start = time.perf_counter()
    cases: list[dict[str, Any]] = []

    def grid(stage: dict[str, Any]) -> list[tuple[int, int, int]]:
        if args.data == "fcon":
            return [(0, r, 0) for r in stage["r"]]
        return list(itertools.product(stage["n"], stage["r"], stage["levels"]))

    if "predict" in args.stages:
        for lik in args.likelihoods:
            for n, r, levels in grid(args.preset["predict"]):
                res = run_predict_case(args.data, lik, n, r, levels, args)
                cases.append(res)
                print(f"done predict {res['case']}", flush=True)
    if "fit" in args.stages:
        with tempfile.TemporaryDirectory(
            prefix="pcnbench_hbr_", dir=args.scratch
        ) as tmp:
            jobs = [
                (lik, n, r, lv, args.preset["fit"]["sampling"])
                for lik in args.likelihoods
                for n, r, lv in grid(args.preset["fit"])
            ]
            if args.preset["default_fit"]:
                jobs.append(("normal", 1000, 1, 2, LIBRARY_DEFAULT_SAMPLING))
            for i, (lik, n, r, lv, sampling) in enumerate(jobs):
                res = run_fit_case(
                    args.data, lik, n, r, lv, sampling, args, Path(tmp) / f"fit{i}"
                )
                cases.append(res)
                print(f"done fit {res['case']}", flush=True)

    params = {k: v for k, v in vars(args).items() if k != "preset"}
    params.update({"grid": args.preset, "wall_time_s": time.perf_counter() - t_start})
    path = write_results("bench_hbr", params, cases, args.out)
    print_table(cases)
    n_fail = sum(not c["passed"] for case in cases for c in case.get("checks", []))
    print(f"determinism checks: {'all passed' if n_fail == 0 else f'{n_fail} FAILED'}")
    print(f"wrote {path}  (total {params['wall_time_s']:.1f} s)")


if __name__ == "__main__":
    main()
