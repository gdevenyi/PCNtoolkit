"""Check that HBR fits agree within MCMC noise (the MCMC tolerance class).

Two MCMC runs never give identical draws, so exact comparison is not
possible. Instead :func:`benchmarks.compare.compare_mcmc` checks, for every
posterior parameter, that the means and SDs of two fits differ by at most
3 x their combined Monte Carlo standard error, that test-set Z-scores agree
within 0.05, and that R-hat <= 1.01 in both fits.

HBR passes no ``random_seed`` to PyMC, so each fit uses a fresh random seed
and is not reproducible. That is what makes two fits "different seeds".

Three ways to use it (run from the repository root):

1. Two fits of the current code, same process::

       python -m benchmarks.check_hbr_accuracy

2. Save a reference fit on the base branch, then check a PR branch against
   it::

       git switch dev
       python -m benchmarks.check_hbr_accuracy \
           --save-reference benchmarks/models/hbr_ref
       git switch my-speedup-branch
       python -m benchmarks.check_hbr_accuracy --reference benchmarks/models/hbr_ref

3. Quick check that the script works (short chains; R-hat, Z and MCSE
   differences are reported but not checked)::

       python -m benchmarks.check_hbr_accuracy --size smoke

Reference problem (``--reference-model simple``, the default): synthetic
data (``benchmarks.data.make_synthetic``, 1 response variable, 1 batch effect
with 2 levels) and a Normal HBR model with 8 posterior parameters. The mean
is a polynomial of degree 2 in age plus a random intercept per batch level
(centered form); the SD is softplus of a straight line in age. ``--size
full`` (default) uses N=1000, 4 chains x 3000 draws and tune=500 (about
25 s). Unchanged code passed 29 of 30 runs; the 3-MCSE rule makes 16 tests
per run, so about 1 run in 25 fails by chance. See ``benchmarks/README.md``.

``--reference-model hard`` is the library default Normal model (B-spline
mean and SD, non-centered random intercept, 21 parameters). It is not the
reference because it often does not converge on unchanged code: one chain
does not mix with the others (in some runs R-hat above 1.01 on most
parameters, up to 1.06-1.10), and 8 of 14 measured runs failed. Use it
only to study that problem.

``--tune``, ``--draws`` and ``--chains`` change the sampling for an
experiment. The exit code is 0 if all checks pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

from benchmarks.bench_hbr import LIBRARY_DEFAULT_SAMPLING, make_hbr, new_model
from benchmarks.compare import MCMC_MAX_RHAT, MCMC_N_MCSE, MCMC_Z_ATOL, compare_mcmc
from benchmarks.data import make_dataset
from benchmarks.env import git_info
from benchmarks.timing import write_results
from pcntoolkit import (
    HBR,
    LinearBasisFunction,
    NormalLikelihood,
    NormativeModel,
    NormData,
    PolynomialBasisFunction,
    make_prior,
)
from pcntoolkit.util.output import Output

REFERENCE_MODELS = ("simple", "hard")

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "n": 200,
        "r": 1,
        "levels": 2,
        "sampling": {"tune": 100, "draws": 100, "chains": 2},
        # 100 draws are too few for reliable R-hat and MCSE values; smoke
        # only tests the plumbing.
        "max_rhat": float("inf"),
        "z_atol": float("inf"),
        "n_mcse": float("inf"),
    },
    "full": {
        "n": 1000,
        "r": 1,
        "levels": 2,
        # Agreed sampling for the MCMC class. With the default reference
        # model, unchanged code passed 29 of 30 runs; see benchmarks/README.md.
        "sampling": {**LIBRARY_DEFAULT_SAMPLING, "draws": 3000, "chains": 4},
        "max_rhat": MCMC_MAX_RHAT,
        "z_atol": MCMC_Z_ATOL,
        "n_mcse": MCMC_N_MCSE,
    },
}


def make_simple_likelihood() -> NormalLikelihood:
    """Normal likelihood of the default reference problem.

    Returns
    -------
    NormalLikelihood
        Mean: polynomial of degree 2 in the covariate plus a random intercept
        per batch level. SD: softplus of a straight line in the covariate (no
        basis functions, no random effects).
    """
    mu = make_prior(
        linear=True,
        slope=make_prior(dist_params=(0.0, 3.0)),
        intercept=make_prior(
            random=True,
            # Each level has hundreds of subjects, so the offsets are well
            # determined; the non-centered form then makes a narrow curved
            # ridge between the group SD and the unit-scale offsets, with
            # hundreds of divergences and R-hat up to 1.5 when tested.
            centered=True,
            mu=make_prior(dist_params=(0.0, 1.0)),
            sigma=make_prior(dist_name="Gamma", dist_params=(1.0, 0.5)),
        ),
        # No constant column: a B-spline basis sums to 1, so its weights and
        # the intercept can trade off against each other.
        basis_function=PolynomialBasisFunction(degree=2),
    )
    sigma = make_prior(
        linear=True,
        slope=make_prior(dist_params=(0.0, 2.0)),
        intercept=make_prior(dist_params=(0.0, 1.0)),
        mapping="softplus",
        mapping_params=(0.0, 2.0),
        basis_function=LinearBasisFunction(),
    )
    return NormalLikelihood(mu, sigma)


def make_reference_hbr(
    reference_model: str,
    likelihood: str,
    tune: int,
    draws: int,
    chains: int,
    sampler: str,
) -> HBR:
    """Build the HBR template of a reference problem.

    Parameters
    ----------
    reference_model : str
        One of ``REFERENCE_MODELS``.
    likelihood : str
        ``"normal"`` or ``"shashb"``; only used by ``"hard"``.
    tune, draws, chains : int
        Sampling settings.
    sampler : str
        NUTS sampler name.

    Returns
    -------
    HBR
        Unfitted template model.
    """
    if reference_model == "hard":
        return make_hbr(likelihood, tune, draws, chains, sampler=sampler)
    return HBR(
        likelihood=make_simple_likelihood(),
        tune=tune,
        draws=draws,
        chains=chains,
        cores=chains,
        nuts_sampler=sampler,
        progressbar=False,
    )


def fit_model(
    train: NormData,
    reference_model: str,
    likelihood: str,
    sampling: dict[str, int],
    sampler: str,
    save_dir: str,
    save: bool,
) -> tuple[NormativeModel, float]:
    """Fit one HBR normative model and return it with its fit time.

    Parameters
    ----------
    train : NormData
        Training data (copied).
    reference_model : str
        One of ``REFERENCE_MODELS``.
    likelihood : str
        ``"normal"`` or ``"shashb"``; only used by ``"hard"``.
    sampling : dict[str, int]
        tune, draws, chains.
    sampler : str
        NUTS sampler name.
    save_dir : str
        Model folder.
    save : bool
        Whether to save the model to ``save_dir``.

    Returns
    -------
    tuple[NormativeModel, float]
        Fitted model and wall time in seconds of ``fit``.
    """
    hbr = make_reference_hbr(reference_model, likelihood, sampler=sampler, **sampling)
    model = new_model(hbr, save_dir, savemodel=save)
    t0 = time.perf_counter()
    model.fit(copy.deepcopy(train))
    return model, time.perf_counter() - t0


def zscores(model: NormativeModel, test: NormData) -> Any:
    """Z-scores of the test set, ordered by response variable name.

    Parameters
    ----------
    model : NormativeModel
        Fitted model.
    test : NormData
        Test data (copied).

    Returns
    -------
    xarray.DataArray
        Z with dims (observations, response_vars).
    """
    d = copy.deepcopy(test)
    model.compute_zscores(d)
    return d["Z"].sortby("response_vars")


def main(argv: list[str] | None = None) -> int:
    """Run the check.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    int
        0 if every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", choices=sorted(PRESETS), default="full")
    parser.add_argument(
        "--reference-model",
        choices=REFERENCE_MODELS,
        default="simple",
        help="reference problem; 'hard' is known to mix poorly (see README)",
    )
    parser.add_argument(
        "--likelihood",
        choices=("normal", "shashb"),
        default="normal",
        help="likelihood of the 'hard' reference model",
    )
    parser.add_argument("--sampler", default="nutpie")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the synthetic data"
    )
    parser.add_argument("--levels", type=int, help="batch levels (default: preset)")
    parser.add_argument(
        "--tune", type=int, help="tuning steps per chain (default: preset)"
    )
    parser.add_argument(
        "--draws", type=int, help="kept draws per chain (default: preset)"
    )
    parser.add_argument(
        "--chains",
        type=int,
        help="number of chains, also the number of cores (default: preset)",
    )
    parser.add_argument(
        "--max-rhat",
        type=float,
        help="largest allowed R-hat per run (default: 1.01; smoke: not checked)",
    )
    parser.add_argument(
        "--n-mcse",
        type=float,
        help="allowed difference in combined MCSE (default: 3; smoke: not checked)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save-reference", help="fit once and save the model here as the reference"
    )
    group.add_argument(
        "--reference", help="fit once and compare with the reference saved here"
    )
    parser.add_argument(
        "--scratch", help="folder for temporary model files (default: system temp)"
    )
    parser.add_argument(
        "--out", help="result JSON path (default: benchmarks/results/...)"
    )
    parser.add_argument(
        "--z-atol",
        type=float,
        help="Z-score tolerance (default: 0.05; smoke: not checked)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    pre = dict(PRESETS[args.size])
    if args.levels is not None:
        pre["levels"] = args.levels
    pre["sampling"] = dict(pre["sampling"])
    for key in ("tune", "draws", "chains"):
        if getattr(args, key) is not None:
            pre["sampling"][key] = getattr(args, key)
    if args.max_rhat is None:
        args.max_rhat = pre["max_rhat"]
    if args.z_atol is None:
        args.z_atol = pre["z_atol"]
    if args.n_mcse is None:
        args.n_mcse = pre["n_mcse"]
    if not args.verbose:
        Output.set_show_messages(False)
        Output.set_show_warnings(False)
        warnings.filterwarnings("ignore")
        for name in ("pymc", "pytensor", "nutpie"):
            logging.getLogger(name).setLevel(logging.ERROR)

    if args.reference_model != "hard" and args.likelihood != "normal":
        parser.error("--likelihood only applies to --reference-model hard")
    train, test = make_dataset(
        "synthetic", pre["n"], pre["r"], pre["levels"], 1, args.seed
    )
    sampling = pre["sampling"]
    fit_times: list[float] = []

    if args.save_reference:
        ref_dir = Path(args.save_reference)
        _, t = fit_model(
            train,
            args.reference_model,
            args.likelihood,
            sampling,
            args.sampler,
            str(ref_dir),
            save=True,
        )
        meta = {
            "git": git_info(),
            "reference_model": args.reference_model,
            "likelihood": args.likelihood,
            "size": args.size,
            "seed": args.seed,
            "levels": pre["levels"],
            "sampler": args.sampler,
            "sampling": sampling,
        }
        (ref_dir / "reference_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"saved reference to {ref_dir} (fit {t:.1f} s)")
        return 0

    if args.reference:
        meta = json.loads((Path(args.reference) / "reference_meta.json").read_text())
        made_with = (
            # References saved before "reference_model" existed used "hard".
            meta.get("reference_model", "hard"),
            meta["likelihood"],
            meta["size"],
            meta["seed"],
            meta.get("levels"),
        )
        this_run = (
            args.reference_model,
            args.likelihood,
            args.size,
            args.seed,
            pre["levels"],
        )
        if made_with != this_run:
            print(
                f"reference was made with {meta}; pass the same "
                "--reference-model/--likelihood/--size/--seed/--levels"
            )
            return 1
        # References saved before "sampling" was recorded have no entry.
        if meta.get("sampling", sampling) != sampling:
            print(
                f"reference was sampled with {meta['sampling']}, this run with "
                f"{sampling}; pass the same --tune/--draws/--chains"
            )
            return 1

    with tempfile.TemporaryDirectory(
        prefix="pcnbench_hbracc_", dir=args.scratch
    ) as tmp:
        cand, t = fit_model(
            train,
            args.reference_model,
            args.likelihood,
            sampling,
            args.sampler,
            str(Path(tmp) / "cand"),
            save=False,
        )
        fit_times.append(t)
        if args.reference:
            ref = NormativeModel.load(args.reference)
        else:
            ref, t = fit_model(
                train,
                args.reference_model,
                args.likelihood,
                sampling,
                args.sampler,
                str(Path(tmp) / "ref"),
                save=False,
            )
            fit_times.append(t)
        z_cand, z_ref = zscores(cand, test), zscores(ref, test)
        checks = []
        for rv in sorted(cand.response_vars):
            res = compare_mcmc(
                cand[rv].idata,
                ref[rv].idata,
                z_cand.sel(response_vars=rv),
                z_ref.sel(response_vars=rv),
                n_mcse=args.n_mcse,
                max_rhat=args.max_rhat,
                z_atol=args.z_atol,
                name=f"mcmc:{rv}",
            )
            checks.append(res)
            print(res)

    params = {**vars(args), "preset": pre}
    cases = [
        {
            "case": {
                "check": "hbr_accuracy",
                "reference_model": args.reference_model,
                "likelihood": args.likelihood,
                **sampling,
            },
            "timings": [],
            "fit_times_s": fit_times,
            "checks": [c.__dict__ for c in checks],
        }
    ]
    path = write_results("check_hbr_accuracy", params, cases, args.out)
    passed = all(c.passed for c in checks)
    print(
        f"{'PASS' if passed else 'FAIL'}; "
        f"fit times {[round(x, 1) for x in fit_times]} s; wrote {path}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
