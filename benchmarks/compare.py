"""Tolerance checks for speed-up PRs, and a small comparison CLI.

A speed-up PR must show two things: the code is faster, and the results did
not change by more than the agreed tolerance. There are three tolerance
classes:

DETERMINISTIC
    Outputs that do not depend on an optimiser path or on MCMC randomness:
    predictions from a fixed saved model (BLR or HBR), likelihood
    forward/backward, SHASH functions, warps, scalers, basis functions,
    evaluator metrics, NormData scaling. Check: ``rtol=1e-10`` and
    ``atol=0`` (a tiny ``atol`` only where a value can be close to 0).
OPTIMISER
    BLR fit results, because a PR may change optimiser internals. Check: the
    converged negative log-likelihood within ``rtol=1e-8`` and the Z-scores
    within ``atol=1e-6``.
MCMC
    HBR fit results. Two MCMC runs never give the same numbers, so we
    compare each parameter's posterior mean and SD against the Monte Carlo
    standard error (MCSE, the expected noise of that estimate from a finite
    number of draws). Check: ``|mean_a - mean_b| <= 3 * sqrt(mcse_a^2 +
    mcse_b^2)`` (same for SD with ``mcse_sd``), Z-scores within
    ``atol=0.05``, and R-hat (a convergence statistic; 1.0 means the chains
    agree) ``<= 1.01`` in both runs. The limits apply to fits with 4 chains
    x 3000 draws (``check_hbr_accuracy --size full``); the measured rate of
    false failures is in ``benchmarks/README.md``.

Note: ZINB ``forward`` (Z-scores) draws random numbers, so it is only
deterministic when you pass the same ``rng``.

CLI usage (run from the repository root)::

    python -m benchmarks.compare timing OLD.json NEW.json
    python -m benchmarks.compare outputs OLD_DIR NEW_DIR --mode deterministic
    python -m benchmarks.compare outputs OLD_DIR NEW_DIR --mode optimiser
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DET_RTOL = 1e-10
OPT_NLL_RTOL = 1e-8
OPT_Z_ATOL = 1e-6
MCMC_N_MCSE = 3.0
MCMC_Z_ATOL = 0.05
MCMC_MAX_RHAT = 1.01
# Absolute floor for DETERMINISTIC values that can be ~0 (Z, logp, centiles
# and Yhat are signed); without it rtol=1e-10 on 1e-9 asks for 1e-19.
NEAR_ZERO_ATOL = 1e-12

# Variables in a saved prediction file that the deterministic check covers.
PREDICTION_VARS = ("Z", "centiles", "logp", "Yhat", "baseline_logp")


@dataclass
class CheckResult:
    """Outcome of one tolerance check.

    Attributes
    ----------
    name : str
        What was compared.
    passed : bool
        True if every value is within tolerance.
    details : dict[str, Any]
        Numbers behind the decision (max errors, failing items).
    """

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        """Raise ``AssertionError`` with the details if the check failed.

        Raises
        ------
        AssertionError
            If ``passed`` is False.
        """
        if not self.passed:
            raise AssertionError(
                f"{self.name} failed: {json.dumps(self.details, default=str)}"
            )

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {json.dumps(self.details, default=str)}"


def _as_array(x: Any) -> np.ndarray:
    """Convert numpy, xarray or dask input to a float numpy array.

    Parameters
    ----------
    x : Any
        Array-like input.

    Returns
    -------
    np.ndarray
        Materialised float array.
    """
    if hasattr(x, "values") and not isinstance(x, np.ndarray):
        x = x.values
    if hasattr(x, "compute"):
        x = x.compute()
    return np.asarray(x, dtype=float)


def _close(a: Any, b: Any, rtol: float, atol: float, name: str) -> CheckResult:
    """Element-wise ``|a - b| <= atol + rtol * |b|``, NaNs must match.

    Parameters
    ----------
    a, b : Any
        Arrays to compare; ``b`` is the reference.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    name : str
        Label for the result.

    Returns
    -------
    CheckResult
        Pass/fail and the largest absolute and relative errors.
    """
    a = _as_array(a)
    b = _as_array(b)
    if a.shape != b.shape:
        return CheckResult(name, False, {"error": f"shape {a.shape} != {b.shape}"})
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    nan_ok = bool(np.array_equal(nan_a, nan_b))
    ok = ~(nan_a | nan_b)
    diff = np.abs(a[ok] - b[ok])
    ref = np.abs(b[ok])
    within = diff <= atol + rtol * ref
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ref > 0, diff / ref, np.where(diff > 0, np.inf, 0.0))
    details = {
        "rtol": rtol,
        "atol": atol,
        "n": int(a.size),
        "n_fail": int((~within).sum()) + (0 if nan_ok else int((nan_a != nan_b).sum())),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "nan_pattern_equal": nan_ok,
    }
    return CheckResult(name, bool(within.all()) and nan_ok, details)


def compare_deterministic(
    a: Any,
    b: Any,
    rtol: float = DET_RTOL,
    atol: float = 0.0,
    name: str = "deterministic",
) -> CheckResult:
    """DETERMINISTIC class: values must agree to ``rtol=1e-10``.

    Parameters
    ----------
    a : Any
        Candidate values (numpy, xarray or dask array).
    b : Any
        Reference values, same shape.
    rtol : float
        Relative tolerance, default 1e-10.
    atol : float
        Absolute tolerance, default 0. Use a tiny value (e.g. 1e-12) only
        for outputs that can be near 0, and say why in a comment.
    name : str
        Label for the result.

    Returns
    -------
    CheckResult
        Pass/fail with max absolute and relative error.
    """
    return _close(a, b, rtol, atol, name)


def compare_optimiser(
    nll_a: Any,
    nll_b: Any,
    z_a: Any,
    z_b: Any,
    nll_rtol: float = OPT_NLL_RTOL,
    z_atol: float = OPT_Z_ATOL,
    name: str = "optimiser",
) -> CheckResult:
    """OPTIMISER class (BLR fit): converged NLL and Z-scores.

    Use ``BLR.nlZ`` after ``fit``. With the default l-bfgs-b optimiser this
    is the value of the minimised objective, so it includes the L1/L2
    penalty on the hyperparameters (``fit`` sets ``nlZ = out[1]`` after it
    recomputes ``loglik``). Compare like with like: both runs must use the
    same optimiser and penalty settings.

    Parameters
    ----------
    nll_a, nll_b : Any
        Negative log-likelihood per response variable (scalar or array).
    z_a, z_b : Any
        Z-scores of the same observations.
    nll_rtol : float
        Relative tolerance on the NLL, default 1e-8.
    z_atol : float
        Absolute tolerance on Z, default 1e-6.
    name : str
        Label for the result.

    Returns
    -------
    CheckResult
        Combined pass/fail with the NLL and Z details.
    """
    nll = _close(nll_a, nll_b, nll_rtol, 0.0, "nll")
    z = _close(z_a, z_b, 0.0, z_atol, "z")
    return CheckResult(
        name, nll.passed and z.passed, {"nll": nll.details, "z": z.details}
    )


def mcmc_summary(idata: Any, var_names: list[str] | None = None) -> Any:
    """Unrounded arviz summary of the posterior group.

    Parameters
    ----------
    idata : Any
        ``xarray.DataTree`` (arviz >= 1.0) or InferenceData with a
        ``posterior`` group.
    var_names : list[str] | None
        Variables to keep; None keeps all.

    Returns
    -------
    pandas.DataFrame
        Columns include ``mean``, ``sd``, ``mcse_mean``, ``mcse_sd`` and
        ``r_hat``; one row per scalar parameter element.
    """
    import arviz as az

    # round_to="none" returns floats; the default rounds and returns strings.
    return az.summary(
        idata, var_names=var_names, group="posterior", kind="all", round_to="none"
    )


def compare_mcmc(
    idata_a: Any,
    idata_b: Any,
    z_a: Any,
    z_b: Any,
    var_names: list[str] | None = None,
    n_mcse: float = MCMC_N_MCSE,
    z_atol: float = MCMC_Z_ATOL,
    max_rhat: float = MCMC_MAX_RHAT,
    name: str = "mcmc",
) -> CheckResult:
    """MCMC class (HBR fit): posterior moments within MCSE, Z, R-hat.

    Parameters
    ----------
    idata_a, idata_b : Any
        Posterior draws of two fits (``HBR.idata``).
    z_a, z_b : Any
        Z-scores of the same observations from each fit.
    var_names : list[str] | None
        Parameters to compare; None compares every posterior variable that
        is in both runs.
    n_mcse : float
        Allowed difference in units of combined MCSE, default 3.
    z_atol : float
        Absolute tolerance on Z, default 0.05.
    max_rhat : float
        Largest allowed R-hat in each run, default 1.01.
    name : str
        Label for the result.

    Returns
    -------
    CheckResult
        Pass/fail with the worst parameters and Z error.

    Notes
    -----
    A parameter with zero spread (constant or fixed) has MCSE 0 and R-hat
    NaN. For such parameters the means must agree within ``1e-12 + 1e-10 *
    |mean|`` and R-hat is not checked.
    """
    sa = mcmc_summary(idata_a, var_names)
    sb = mcmc_summary(idata_b, var_names)
    common = [i for i in sa.index if i in sb.index]
    missing = sorted(set(sa.index).symmetric_difference(sb.index))
    sa, sb = sa.loc[common], sb.loc[common]

    fails: list[dict[str, Any]] = []
    worst_ratio = 0.0
    worst_where: tuple[str, str] | None = None
    for stat, mcse in (("mean", "mcse_mean"), ("sd", "mcse_sd")):
        diff = np.abs(sa[stat].to_numpy(float) - sb[stat].to_numpy(float))
        comb = np.sqrt(sa[mcse].to_numpy(float) ** 2 + sb[mcse].to_numpy(float) ** 2)
        constant = ~np.isfinite(comb) | (comb == 0)
        limit = np.where(
            constant, 1e-12 + 1e-10 * np.abs(sb[stat].to_numpy(float)), n_mcse * comb
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(constant, 0.0, diff / comb)
        if ratio.size and float(np.nanmax(ratio)) > worst_ratio:
            worst_ratio = float(np.nanmax(ratio))
            worst_where = (common[int(np.nanargmax(ratio))], stat)
        for idx in np.flatnonzero(diff > limit):
            fails.append(
                {
                    "param": common[idx],
                    "stat": stat,
                    "diff": float(diff[idx]),
                    "limit": float(limit[idx]),
                }
            )

    rhat_fail = {}
    for label, s in (("a", sa), ("b", sb)):
        rhat = s["r_hat"].to_numpy(float)
        varying = np.isfinite(s["mcse_mean"].to_numpy(float)) & (
            s["sd"].to_numpy(float) > 0
        )
        bad = varying & ~(rhat <= max_rhat)
        rhat_fail[label] = {
            "max_r_hat": float(np.nanmax(rhat[varying])) if varying.any() else None,
            "n_over": int(bad.sum()),
            "params_over": [common[i] for i in np.flatnonzero(bad)][:20],
        }

    z = _close(z_a, z_b, 0.0, z_atol, "z")
    passed = (
        not fails
        and not missing
        and z.passed
        and all(v["n_over"] == 0 for v in rhat_fail.values())
    )
    details = {
        "n_params": len(common),
        "missing_params": missing[:20],
        "worst_diff_in_mcse": worst_ratio,
        "worst_param": worst_where[0] if worst_where else None,
        "worst_stat": worst_where[1] if worst_where else None,
        "moment_failures": fails[:20],
        "n_moment_failures": len(fails),
        "r_hat": rhat_fail,
        "z": z.details,
    }
    return CheckResult(name, passed, details)


# ---------------------------------------------------------------- CLI ----


def _timing_index(path: Path) -> dict[tuple[str, str], float]:
    """Map (case label, operation) to median seconds for one result file.

    Parameters
    ----------
    path : Path
        JSON file written by a benchmark script.

    Returns
    -------
    dict[tuple[str, str], float]
        Median time per (case, operation).
    """
    payload = json.loads(path.read_text())
    out = {}
    for c in payload["cases"]:
        label = " ".join(f"{k}={v}" for k, v in c["case"].items())
        for t in c["timings"]:
            out[(label, t["name"])] = t["median"]
    return out


def cli_timing(old: Path, new: Path) -> int:
    """Print the speed-up of ``new`` over ``old`` for matching timings.

    Parameters
    ----------
    old : Path
        Reference result JSON.
    new : Path
        Candidate result JSON.

    Returns
    -------
    int
        Exit code (0).
    """
    a, b = _timing_index(old), _timing_index(new)
    for name, p in (("old", old), ("new", new)):
        git = json.loads(p.read_text())["env"]["git"]
        print(
            f"{name}: {p}  commit={git['commit']} "
            f"dirty_pcntoolkit={git['dirty_pcntoolkit']}"
        )
    rows = [("case", "operation", "old_s", "new_s", "speedup")]
    for key in a:
        if key in b:
            speed = a[key] / b[key] if b[key] > 0 else float("inf")
            rows.append(
                (key[0], key[1], f"{a[key]:.4g}", f"{b[key]:.4g}", f"{speed:.2f}x")
            )
    only = set(a).symmetric_difference(b)
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)))
    if only:
        print(f"{len(only)} timings are only in one file and are not shown.")
    return 0


def _load_nc(path: Path) -> Any:
    """Open a saved NormData NetCDF file.

    Parameters
    ----------
    path : Path
        ``.nc`` file written by ``NormData.to_netcdf``.

    Returns
    -------
    xarray.Dataset
        Loaded dataset.
    """
    import xarray as xr

    return xr.load_dataset(path)


def _align(a: Any, b: Any) -> Any:
    """Reorder DataArray ``a`` to the dimension order and labels of ``b``.

    Parameters
    ----------
    a : xarray.DataArray
        Candidate values.
    b : xarray.DataArray
        Reference values.

    Returns
    -------
    xarray.DataArray
        ``a`` with the dims and index labels of ``b``.
    """
    a = a.transpose(*b.dims)
    return a.sel({d: b[d].values for d in b.dims if d in b.indexes})


def cli_outputs(old: Path, new: Path, mode: str) -> int:
    """Compare saved prediction outputs of two runs.

    Each directory holds ``<case>.nc`` files (NormData with Z, centiles,
    logp, Yhat) and, for BLR, ``<case>_nll.json`` with the NLL per response
    variable. Benchmark scripts write these with ``--save-outputs``.

    Parameters
    ----------
    old : Path
        Reference directory.
    new : Path
        Candidate directory.
    mode : str
        ``"deterministic"`` or ``"optimiser"``.

    Returns
    -------
    int
        0 if every check passed, 1 otherwise.
    """
    results: list[CheckResult] = []
    files = sorted(p.name for p in old.glob("*.nc"))
    if not files:
        print(f"No .nc files in {old}")
        return 1
    for fname in files:
        if not (new / fname).exists():
            results.append(CheckResult(fname, False, {"error": "missing in new"}))
            continue
        a, b = _load_nc(new / fname), _load_nc(old / fname)
        stem = fname[:-3]
        if mode == "deterministic":
            for var in PREDICTION_VARS:
                if var in b:
                    results.append(
                        compare_deterministic(
                            _align(a[var], b[var]),
                            b[var],
                            atol=NEAR_ZERO_ATOL,
                            name=f"{stem}:{var}",
                        )
                    )
        else:
            nll_a = json.loads((new / f"{stem}_nll.json").read_text())
            nll_b = json.loads((old / f"{stem}_nll.json").read_text())
            keys = sorted(nll_b)
            results.append(
                compare_optimiser(
                    [nll_a[k] for k in keys],
                    [nll_b[k] for k in keys],
                    _align(a["Z"], b["Z"]),
                    b["Z"],
                    name=f"{stem}:optimiser",
                )
            )
    for r in results:
        print(r)
    n_fail = sum(not r.passed for r in results)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; default ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_t = sub.add_parser("timing", help="speed-up table of two benchmark JSON files")
    p_t.add_argument("old", type=Path)
    p_t.add_argument("new", type=Path)
    p_o = sub.add_parser(
        "outputs", help="tolerance check of two --save-outputs directories"
    )
    p_o.add_argument("old", type=Path)
    p_o.add_argument("new", type=Path)
    p_o.add_argument(
        "--mode", choices=("deterministic", "optimiser"), default="deterministic"
    )
    args = parser.parse_args(argv)
    if args.cmd == "timing":
        return cli_timing(args.old, args.new)
    return cli_outputs(args.old, args.new, args.mode)


if __name__ == "__main__":
    sys.exit(main())
