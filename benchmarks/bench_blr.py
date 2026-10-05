"""Benchmark BLR normative models: fit and the predict-side steps.

Operations timed per case (case = data source, N, R, batch levels, config):

fit_only
    The regression fits only: ``register_data_info``, ``preprocess`` and
    ``BLR.fit`` for every response variable. This is the same loop as the
    start of ``NormativeModel.fit``.
fit_total
    ``NormativeModel.fit`` with ``savemodel``, ``saveresults``,
    ``saveplots`` and ``evaluate_model`` all False. ``fit`` always ends with
    ``predict`` on the training data, so this is fit_only plus one full
    predict pass (Z, centiles, baseline logp, logp, Yhat). Nothing is written
    to disk except empty result/plot folders under ``--scratch``.
load
    ``NormativeModel.load`` of the saved model.
compute_zscores / compute_centiles / compute_logp / compute_yhat
    Each step alone, on the test set, from the loaded model.
    ``compute_centiles`` uses ``recompute=True`` so every repeat does work.
predict
    ``NormativeModel.predict`` on the test set with saving, plotting and
    evaluation off.
evaluate
    ``Evaluator.evaluate`` on the predicted test set (all metrics).

Each case also checks that two loads of the same saved model give the same
predictions (DETERMINISTIC class, ``rtol=1e-10``); the result is in the
JSON under ``checks``.

Run from the repository root::

    python -m benchmarks.bench_blr --size smoke
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

from benchmarks.compare import PREDICTION_VARS, compare_deterministic
from benchmarks.data import make_dataset
from benchmarks.timing import Timing, print_table, time_call, write_results
from pcntoolkit import BLR, BsplineBasisFunction, NormativeModel, NormData
from pcntoolkit.util.output import Output

# Sizes per preset: N (train and test each), R (response vars), batch levels.
# FCON1000 has N=1078 and 2+23 batch levels; its "fcon_*" keys override the
# synthetic grid because a heteroskedastic fit with 23 sites is ~10 s per
# response variable.
PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "n": [500],
        "r": [2],
        "levels": [2],
        "configs": ["plain", "hetero", "warp"],
        "repeats": 2,
        "warmup": 1,
        "fcon_r": [1],
        "fcon_configs": ["plain", "hetero"],
        "fcon_repeats": 1,
    },
    "small": {
        "n": [1000],
        "r": [1, 10],
        "levels": [2, 20],
        "configs": ["plain", "hetero", "warp"],
        "repeats": 3,
        "warmup": 1,
    },
    "medium": {
        "n": [1000, 10000],
        "r": [1, 10],
        "levels": [2, 20],
        "configs": ["plain", "hetero", "warp"],
        "repeats": 3,
        "warmup": 1,
    },
    "large": {
        "n": [1000, 10000, 50000],
        "r": [1, 10, 100],
        "levels": [2, 20],
        "configs": ["hetero", "warp"],
        "repeats": 2,
        "warmup": 1,
    },
}


def make_blr(config: str) -> BLR:
    """Build the BLR template for a named configuration.

    Parameters
    ----------
    config : str
        ``"plain"`` (B-spline mean, constant noise, batch fixed effect),
        ``"hetero"`` (plus noise that changes with the covariate and a batch
        fixed effect on the noise) or ``"warp"`` (hetero plus a
        sinh-arcsinh warp of Y).

    Returns
    -------
    BLR
        Unfitted template model; optimiser is the default L-BFGS-B.

    Raises
    ------
    ValueError
        If ``config`` is not known.
    """
    common = {
        "basis_function_mean": BsplineBasisFunction(degree=3, nknots=5),
        "fixed_effect": True,
    }
    if config == "plain":
        return BLR(**common)
    hetero = {
        **common,
        "heteroskedastic": True,
        "fixed_effect_var": True,
        "basis_function_var": BsplineBasisFunction(degree=3, nknots=5),
    }
    if config == "hetero":
        return BLR(**hetero)
    if config == "warp":
        return BLR(**hetero, warp_name="WarpSinhArcsinh")
    raise ValueError(f"Unknown BLR config {config!r}")


def new_model(config: str, save_dir: str) -> NormativeModel:
    """Make a NormativeModel with all saving, plotting and evaluation off.

    Parameters
    ----------
    config : str
        BLR configuration name, see :func:`make_blr`.
    save_dir : str
        Folder the model may create sub-folders in.

    Returns
    -------
    NormativeModel
        Unfitted model.
    """
    return NormativeModel(
        make_blr(config),
        savemodel=False,
        evaluate_model=False,
        saveresults=False,
        saveplots=False,
        save_dir=save_dir,
    )


def fit_regressions_only(model: NormativeModel, data: NormData) -> None:
    """Run only the regression-fit part of ``NormativeModel.fit``.

    Parameters
    ----------
    model : NormativeModel
        Fresh unfitted model.
    data : NormData
        Training data; it is scaled in place.
    """
    model.register_data_info(data)
    model.preprocess(data)
    for rv in model.response_vars:
        X, be, be_maps, Y, _ = model.extract_data(data.sel({"response_vars": rv}))
        model[rv].fit(X, be, be_maps, Y)


def predict_outputs(model: NormativeModel, data: NormData) -> NormData:
    """Run the predict steps without saving and return the data.

    Parameters
    ----------
    model : NormativeModel
        Fitted model.
    data : NormData
        Test data; a deep copy is used.

    Returns
    -------
    NormData
        Copy of the data with Z, centiles, baseline_logp, logp and Yhat.
    """
    d = copy.deepcopy(data)
    model.compute_zscores(d)
    model.compute_centiles(d, recompute=True)
    model.compute_baseline_logp(d)
    model.compute_logp(d)
    model.compute_yhat(d)
    return d


def run_case(
    source: str,
    n: int,
    r: int,
    levels: int,
    config: str,
    args: argparse.Namespace,
    scratch: Path,
) -> dict[str, Any]:
    """Time one case and run the determinism check.

    Parameters
    ----------
    source : str
        ``"synthetic"`` or ``"fcon"``.
    n : int
        Observations per set.
    r : int
        Response variables.
    levels : int
        Batch levels per batch-effect dimension.
    config : str
        BLR configuration name.
    args : argparse.Namespace
        Parsed command line.
    scratch : Path
        Scratch folder for this case.

    Returns
    -------
    dict[str, Any]
        Case parameters, timings and checks.
    """
    train, test = make_dataset(source, n, r, levels, args.batch_dims, args.seed)
    case = {
        "data": source,
        "n_train": int(train.X.shape[0]),
        "n_test": int(test.X.shape[0]),
        "r": r,
        "levels": levels if source == "synthetic" else "fcon",
        "config": config,
    }
    rep, warm = args.repeats, args.warmup
    timings: list[Timing] = []
    state: dict[str, Any] = {}

    def fresh() -> None:
        state["model"] = new_model(config, str(scratch / "fit"))
        state["data"] = copy.deepcopy(train)

    timings.append(
        time_call(
            "fit_only",
            lambda: fit_regressions_only(state["model"], state["data"]),
            rep,
            warm,
            fresh,
        )
    )
    timings.append(
        time_call(
            "fit_total", lambda: state["model"].fit(state["data"]), rep, warm, fresh
        )
    )

    fitted: NormativeModel = state["model"]
    model_dir = scratch / "model"
    fitted.save(str(model_dir))
    nll = {rv: float(fitted[rv].nlZ) for rv in fitted.response_vars}

    timings.append(
        time_call("load", lambda: NormativeModel.load(str(model_dir)), rep, warm)
    )
    loaded = NormativeModel.load(str(model_dir))
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
    timings.append(
        time_call("evaluate", lambda: loaded.evaluator.evaluate(d), rep, warm)
    )

    # Two independent loads of one saved model must predict the same values.
    out_a = predict_outputs(NormativeModel.load(str(model_dir)), test)
    out_b = predict_outputs(NormativeModel.load(str(model_dir)), test)
    checks = [
        compare_deterministic(out_a[v], out_b[v], name=f"reload_{v}").__dict__
        for v in PREDICTION_VARS
        if v in out_b
    ]

    if args.save_outputs:
        outdir = Path(args.save_outputs)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f"blr_{source}_n{n}_r{r}_l{levels}_{config}"
        out_a.to_netcdf(str(outdir / f"{stem}.nc"))
        (outdir / f"{stem}_nll.json").write_text(json.dumps(nll, indent=2))

    return {"case": case, "timings": timings, "nll": nll, "checks": checks}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run all cases, write JSON and print a table.

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
        "--configs",
        nargs="+",
        choices=("plain", "hetero", "warp"),
        help="override preset configs",
    )
    parser.add_argument(
        "--batch-dims", type=int, default=1, help="batch-effect dimensions (synthetic)"
    )
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
        "--save-outputs", help="write test predictions (.nc) and NLL (.json) here"
    )
    parser.add_argument(
        "--out", help="result JSON path (default: benchmarks/results/...)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="show pcntoolkit messages and warnings"
    )
    args = parser.parse_args(argv)
    preset = PRESETS[args.size]
    fcon = args.data == "fcon"
    default_repeats = (
        preset.get("fcon_repeats", preset["repeats"]) if fcon else preset["repeats"]
    )
    args.repeats = default_repeats if args.repeats is None else args.repeats
    args.warmup = preset["warmup"] if args.warmup is None else args.warmup
    configs = args.configs or (
        preset.get("fcon_configs", preset["configs"]) if fcon else preset["configs"]
    )
    if not args.verbose:
        Output.set_show_messages(False)
        Output.set_show_warnings(False)
        warnings.filterwarnings("ignore")

    if fcon:
        # FCON1000 has a fixed N and fixed batch effects; only R varies.
        grid = [
            (0, r, 0, c) for r in preset.get("fcon_r", preset["r"]) for c in configs
        ]
    else:
        grid = list(
            itertools.product(preset["n"], preset["r"], preset["levels"], configs)
        )

    t_start = time.perf_counter()
    cases = []
    with tempfile.TemporaryDirectory(prefix="pcnbench_blr_", dir=args.scratch) as tmp:
        for i, (n, r, levels, config) in enumerate(grid):
            case_dir = Path(tmp) / f"case{i}"
            res = run_case(args.data, n, r, levels, config, args, case_dir)
            cases.append(res)
            print(f"done {i + 1}/{len(grid)}: {res['case']}", flush=True)
    params = {
        **vars(args),
        "configs": configs,
        "grid": preset,
        "wall_time_s": time.perf_counter() - t_start,
    }
    path = write_results("bench_blr", params, cases, args.out)
    print_table(cases)
    n_fail = sum(not c["passed"] for case in cases for c in case["checks"])
    print(f"determinism checks: {'all passed' if n_fail == 0 else f'{n_fail} FAILED'}")
    print(f"wrote {path}  (total {params['wall_time_s']:.1f} s)")


if __name__ == "__main__":
    main()
