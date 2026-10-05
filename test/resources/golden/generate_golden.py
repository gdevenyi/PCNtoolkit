"""Generate the golden-value files used by ``test/test_golden``.

Run from the repository root:

    .venv/bin/python test/resources/golden/generate_golden.py [--out DIR]

Without ``--out`` the files go next to this script. Run it only on the code
whose behaviour you want to pin (normally ``dev`` before a speedup). See
``README.md`` in this directory for the rules.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
# Make ``test.test_golden._common`` importable when run as a script.
sys.path.insert(0, str(REPO_ROOT))

import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor  # noqa: E402
import scipy  # noqa: E402
import xarray as xr  # noqa: E402

import pcntoolkit  # noqa: E402
from test.test_golden import _common as c  # noqa: E402

# Environment variables that change BLAS/OpenMP threading. They are recorded,
# never set.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
)


@contextmanager
def seeded_pm_sample(seed: int) -> Iterator[None]:
    """Make ``pm.sample`` reproducible while the block runs.

    ``HBR.fit`` calls ``pm.sample`` without a ``random_seed``. This wraps it so
    every call in the block uses ``seed``.

    Parameters
    ----------
    seed : int
        Seed passed as ``random_seed``.

    Yields
    ------
    None
    """
    original = pm.sample
    pm.sample = functools.partial(original, random_seed=seed)
    try:
        yield
    finally:
        pm.sample = original


def child_seed(name: str) -> int:
    """Derive a stable seed for one section from the master seed.

    Parameters
    ----------
    name : str
        Section name.

    Returns
    -------
    int
        A seed that depends only on ``c.SEED`` and ``name``.
    """
    return int(np.random.SeedSequence([c.SEED, *name.encode()]).generate_state(1)[0])


def save_npz(out_dir: Path, name: str, arrays: dict[str, Any]) -> None:
    """Write arrays to ``<out_dir>/<name>.npz`` (compressed).

    Parameters
    ----------
    out_dir : Path
        Output directory.
    name : str
        File name without extension.
    arrays : dict[str, Any]
        Arrays to store. Keys are sorted so the file layout is stable.
    """
    np.savez_compressed(
        out_dir / f"{name}.npz", **{k: np.asarray(arrays[k]) for k in sorted(arrays)}
    )


def save_model(model: c.NormativeModel, out_dir: Path, name: str) -> None:
    """Save a fitted model with paths relative to ``out_dir``.

    The JSON files store ``save_dir`` and ``idata_path``. Saving from inside
    ``out_dir`` with a relative path keeps those free of machine paths.

    Parameters
    ----------
    model : c.NormativeModel
        Fitted model.
    out_dir : Path
        Golden directory.
    name : str
        Model directory name.
    """
    target = out_dir / name
    if target.exists():
        shutil.rmtree(target)
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        # Bypass the save_dir setter, which creates directories.
        model._save_dir = name
        model.save(name)
    finally:
        os.chdir(cwd)
    # save() also creates empty results/ and plots/ folders; drop them.
    for sub in ("results", "plots"):
        sub_dir = target / sub
        if sub_dir.exists() and not any(sub_dir.iterdir()):
            sub_dir.rmdir()


def disjoint_union(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """Merge inputs and outputs, refusing a key that is in both.

    A shared key would make the saved output overwrite the saved input.

    Parameters
    ----------
    inputs : dict[str, Any]
        Input arrays.
    outputs : dict[str, Any]
        Output arrays.

    Returns
    -------
    dict[str, Any]
        Both dicts merged.

    Raises
    ------
    KeyError
        If a key is in both dicts.
    """
    shared = inputs.keys() & outputs.keys()
    if shared:
        raise KeyError(f"Input and output keys overlap: {sorted(shared)}")
    return inputs | outputs


def make_math(out_dir: Path) -> None:
    """Pin shash helpers, likelihood numpy maps and warps.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    """
    rng = np.random.default_rng(child_seed("math"))
    n_obs, n_samples = 40, 3
    inp: dict[str, np.ndarray] = {
        "shash_x": np.linspace(-6.0, 6.0, 41),
        "shash_eps": np.array([-1.0, 0.0, 0.7]),
        "shash_delta": np.array([0.5, 1.0, 2.3]),
        "shash_p": np.array([0.1, 0.5, 1.0, 1.7, 3.0, 6.5]),
        "shash_q": rng.uniform(0.2, 4.0, size=(4, 5)),
        "shash_eps2d": rng.uniform(-1.5, 1.5, size=(3, 4)),
        "shash_delta2d": rng.uniform(0.4, 2.5, size=(3, 4)),
        "lik_mu": rng.normal(0.0, 1.0, size=(n_obs, n_samples)),
        "lik_sigma": rng.uniform(0.3, 2.0, size=(n_obs, n_samples)),
        "lik_eps": rng.uniform(-1.0, 1.0, size=(n_obs, n_samples)),
        "lik_delta": rng.uniform(0.5, 2.0, size=(n_obs, n_samples)),
        "lik_Y": rng.normal(0.0, 2.0, size=(n_obs, 1)),
        "lik_Z": rng.normal(0.0, 1.5, size=(n_obs, 1)),
        "lik_Y_unit": rng.uniform(0.01, 0.99, size=(n_obs, 1)),
        "lik_beta_alpha": rng.uniform(0.5, 5.0, size=(n_obs, n_samples)),
        "lik_beta_beta": rng.uniform(0.5, 5.0, size=(n_obs, n_samples)),
        "lik_Y_count": rng.integers(0, 15, size=(n_obs, 1)).astype(float),
        "lik_zinb_mu": rng.uniform(0.5, 8.0, size=(n_obs, n_samples)),
        "lik_zinb_alpha": rng.uniform(0.5, 4.0, size=(n_obs, n_samples)),
        "lik_zinb_psi": rng.uniform(0.3, 0.95, size=(n_obs, n_samples)),
        "lik_zinb_seed": np.array(child_seed("zinb_forward")),
        "warp_x": rng.uniform(0.2, 6.0, size=50),
        "warp_param_Log": np.array([]),
        "warp_param_Affine": np.array([0.3, -0.4]),
        "warp_param_BoxCox": np.array([-0.35]),
        "warp_param_SinhArcsinh": np.array([0.4, 0.25]),
        "warp_param_Compose": np.array([0.3, -0.4, 0.4, 0.25]),
    }
    out = c.shash_outputs(inp) | c.likelihood_outputs(inp) | c.warp_outputs(inp)
    save_npz(out_dir, "math", disjoint_union(inp, out))


def make_shash_chunks(out_dir: Path) -> None:
    """Pin K, P and m1m2 on inputs that span more than one dask chunk.

    The inputs are built by ``c.shash_chunk_inputs`` and are not stored.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    """
    save_npz(out_dir, "shash_chunks", c.shash_chunk_outputs())


def make_transforms(out_dir: Path) -> None:
    """Pin scalers and basis functions.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    """
    rng = np.random.default_rng(child_seed("transforms"))
    train = np.column_stack(
        [rng.uniform(8, 85, 150), rng.normal(0, 1, 150), rng.lognormal(0, 1, 150)]
    )
    # The held-out rows reach past the training range to test extrapolation.
    test = np.column_stack(
        [rng.uniform(0, 95, 60), rng.normal(0, 1.5, 60), rng.lognormal(0, 1.2, 60)]
    )
    inp = {
        "scaler_train": train,
        "scaler_test": test,
        "basis_train": train[:, :2],
        "basis_test": test[:, :2],
    }
    out = c.scaler_outputs(inp) | c.basis_outputs(inp)
    save_npz(out_dir, "transforms", disjoint_union(inp, out))


def make_normdata_eval(out_dir: Path) -> None:
    """Pin NormData scaling, Evaluator metrics and batch-effect helpers.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    """
    rng = np.random.default_rng(child_seed("normdata_eval"))
    n = 180
    ds = c.make_dataset_arrays(rng, n, "gaussian")
    Y = np.column_stack([ds["Y"][:, 0], 3.0 * ds["Y"][:, 0] + rng.normal(0, 1, n)])
    Yhat = Y + rng.normal(0, 0.5, size=Y.shape)
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    # Centiles: per-subject quantiles of a shifted normal around Yhat.
    spread = rng.uniform(0.5, 1.5, size=Y.shape)
    centiles = (
        Yhat[None]
        + spread[None] * np.array([-1.645, -0.674, 0.0, 0.674, 1.645])[:, None, None]
    )
    inp: dict[str, np.ndarray] = {
        "nd_X": ds["X"],
        "nd_be": ds["be"],
        "nd_Y": Y,
        "nd_Yhat": Yhat,
        # A mildly skewed Z so skew/kurtosis/ShapiroW are not at their limits.
        "nd_Z": rng.standard_normal(Y.shape) + 0.2 * rng.standard_exponential(Y.shape),
        "nd_logp": -0.5 * np.log(2 * np.pi) - 0.5 * rng.chisquare(1, Y.shape),
        "nd_baseline_logp": -0.5 * np.log(2 * np.pi)
        - 0.5 * rng.chisquare(1, Y.shape)
        - 0.3,
        "nd_centiles": centiles,
        "nd_centile_levels": levels,
    }
    # One site/sex pair never occurs, and "s3" is a level with no subjects.
    ibc_be = ds["be"].copy()
    ibc_be[(ibc_be[:, 0] == "s2") & (ibc_be[:, 1] == "M"), 1] = "F"
    inp |= {
        "ibc_be": ibc_be,
        "ibc_levels_site": np.array(["s0", "s1", "s2", "s3"]),
        "ibc_levels_sex": np.array(["F", "M"]),
        "ear_in_global": rng.normal(size=7),
        "ear_in_per_subject": rng.normal(size=(5, 7)),
    }
    out = (
        c.normdata_scaling_outputs(inp)
        | c.evaluator_outputs(inp)
        | c.batch_combination_outputs(inp)
        | c.extract_and_reshape_outputs(inp)
    )
    save_npz(out_dir, "normdata_eval", disjoint_union(inp, out))


def make_blr(out_dir: Path, work_dir: Path) -> None:
    """Fit, save and pin the BLR models.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    work_dir : Path
        Scratch directory for files that ``fit`` writes.
    """
    rng = np.random.default_rng(child_seed("blr"))
    train = c.make_dataset_arrays(rng, c.N_TRAIN, "gaussian")
    test = c.make_dataset_arrays(rng, c.N_TEST, "gaussian")
    arrays: dict[str, Any] = {f"train_{k}": v for k, v in train.items()} | {
        f"test_{k}": v for k, v in test.items()
    }

    # Fixed-hyperparameter evaluations use standardised inputs, like a fit would.
    def standardise(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
        return (a - ref.mean(axis=0)) / ref.std(axis=0)

    fx = {
        "X_train": standardise(train["X"], train["X"]),
        "be_train": c.encode_be(train["be"]),
        "Y_train": standardise(train["Y"], train["Y"])[:, 0],
        "X_test": standardise(test["X"], train["X"]),
        "be_test": c.encode_be(test["be"]),
        "Y_test": standardise(test["Y"], train["Y"])[:, 0],
        "Z_test": rng.normal(0, 1.2, c.N_TEST),
    }
    arrays |= {f"fx_{k}": v for k, v in fx.items()}

    for config in c.BLR_CONFIGS:
        n_hyp = c.blr_n_hyp(config, fx["X_train"], fx["be_train"])
        hyp = rng.uniform(-0.5, 0.5, n_hyp)
        arrays[f"fx_{config}_hyp"] = hyp
        for k, v in c.blr_fixed_hyp_outputs(config, hyp, **fx).items():
            arrays[f"fx_{config}_{k}"] = v

        model = c.make_normative_model(
            c.blr_template(config), str(work_dir / f"blr_{config}")
        )
        model.fit(c.to_normdata("train", train["X"], train["be"], train["Y"]))
        blr = model.regression_models[c.RESPONSE_VAR]
        arrays[f"fit_{config}_hyp"] = blr.hyp
        arrays[f"fit_{config}_nlZ"] = np.asarray(blr.nlZ)
        save_model(model, out_dir, f"blr_{config}")

        # Predict from the saved copy, exactly as the tests do.
        loaded = c.NormativeModel.load(str(out_dir / f"blr_{config}"))
        for k, v in c.predict_outputs(
            loaded, c.to_normdata("test", test["X"], test["be"], test["Y"])
        ).items():
            arrays[f"pred_{config}_{k}"] = v
        if config == "plain":
            for k, v in c.sampling_outputs(loaded, child_seed("sampling"), 60).items():
                arrays[f"plain_{k}"] = v
            arrays["plain_sampling_seed"] = np.array(child_seed("sampling"))
    save_npz(out_dir, "blr", arrays)


def make_hbr(out_dir: Path, work_dir: Path) -> dict[str, float]:
    """Fit, save and pin one tiny HBR model per likelihood.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    work_dir : Path
        Scratch directory for files that ``fit`` writes.

    Returns
    -------
    dict[str, float]
        Largest R-hat per likelihood (informational only; the predict outputs
        are pinned whether or not the short chains converged).
    """
    arrays: dict[str, Any] = {}
    rhat: dict[str, float] = {}
    for name in c.HBR_LIKELIHOODS:
        rng = np.random.default_rng(child_seed(f"hbr_{name}"))
        kind = c.hbr_dataset_kind(name)
        train = c.make_dataset_arrays(rng, c.N_TRAIN, kind)
        test = c.make_dataset_arrays(rng, c.N_TEST, kind)
        for k, v in train.items():
            arrays[f"{name}_train_{k}"] = v
        for k, v in test.items():
            arrays[f"{name}_test_{k}"] = v
        template = c.HBR(
            name="golden",
            likelihood=c.hbr_likelihood(name),
            draws=c.HBR_DRAWS,
            tune=c.HBR_TUNE,
            chains=c.HBR_CHAINS,
            cores=1,
            nuts_sampler="pymc",
            progressbar=False,
        )
        model = c.make_normative_model(
            template, str(work_dir / f"hbr_{name}"), outscaler=c.hbr_outscaler(name)
        )
        with seeded_pm_sample(child_seed(f"hbr_{name}_sample")):
            model.fit(c.to_normdata("train", train["X"], train["be"], train["Y"]))
        save_model(model, out_dir, f"hbr_{name}")
        posterior = model.regression_models[c.RESPONSE_VAR].idata["posterior"].dataset
        rhat[name] = float(az.rhat(posterior).to_array().max())

        # Predict from the saved copy, exactly as the tests do.
        loaded = c.NormativeModel.load(str(out_dir / f"hbr_{name}"))
        test_data = c.to_normdata("test", test["X"], test["be"], test["Y"])
        # ZINB Z-scores use an unseeded rng inside forward, so they are not pinned.
        outputs = c.predict_outputs(loaded, test_data, zscores=name != "ZINB")
        for k, v in outputs.items():
            arrays[f"{name}_pred_{k}"] = v
    save_npz(out_dir, "hbr", arrays)
    return rhat


def make_multi_response_arrays(
    rng: np.random.Generator, n: int
) -> dict[str, np.ndarray]:
    """Draw one dataset with ``c.N_MULTI_RESPONSE_VARS`` response variables.

    The columns have clearly different means, scales and age trends, so a
    model or output that is matched to the wrong response variable gives
    clearly wrong numbers.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator.
    n : int
        Number of subjects.

    Returns
    -------
    dict[str, np.ndarray]
        ``X`` (n, 2), ``be`` (n, 2) strings and ``Y`` (n, 3).
    """
    ds = c.make_dataset_arrays(rng, n, "gaussian")
    y, age = ds["Y"][:, 0], ds["X"][:, 0]
    ds["Y"] = np.column_stack(
        [
            y,
            10.0 - 2.0 * y + rng.normal(0.0, 0.3, n),
            3.0 * y + 0.05 * age + rng.normal(0.0, 1.0, n),
        ]
    )
    return ds


def make_multi(out_dir: Path, work_dir: Path) -> None:
    """Fit, save and pin BLR and HBR models with several response variables.

    Every output is stored per response variable name
    (``<key>__<response var>``), so the tests can match them by name.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    work_dir : Path
        Scratch directory for files that ``fit`` writes.
    """
    rng = np.random.default_rng(child_seed("multi"))
    train = make_multi_response_arrays(rng, c.N_TRAIN)
    test = make_multi_response_arrays(rng, c.N_TEST)
    arrays: dict[str, Any] = {f"train_{k}": v for k, v in train.items()} | {
        f"test_{k}": v for k, v in test.items()
    }
    train_data = c.to_normdata("train", train["X"], train["be"], train["Y"])
    arrays["response_vars"] = np.asarray(train_data.response_vars.values, dtype=str)

    blr = c.make_normative_model(c.blr_template("plain"), str(work_dir / "blr_multi"))
    blr.fit(train_data)
    for rv, reg in blr.regression_models.items():
        arrays[f"blr_fit_nlZ__{rv}"] = np.asarray(reg.nlZ)
    save_model(blr, out_dir, "blr_multi")

    template = c.HBR(
        name="golden",
        likelihood=c.hbr_likelihood("Normal"),
        draws=c.HBR_DRAWS,
        tune=c.HBR_TUNE,
        chains=c.HBR_CHAINS,
        cores=1,
        nuts_sampler="pymc",
        progressbar=False,
    )
    hbr = c.make_normative_model(template, str(work_dir / "hbr_multi"))
    with seeded_pm_sample(child_seed("hbr_multi_sample")):
        hbr.fit(c.to_normdata("train", train["X"], train["be"], train["Y"]))
    save_model(hbr, out_dir, "hbr_multi")

    # Predict from the saved copies, exactly as the tests do.
    for kind in ("blr", "hbr"):
        loaded = c.NormativeModel.load(str(out_dir / f"{kind}_multi"))
        test_data = c.to_normdata("test", test["X"], test["be"], test["Y"])
        for key, per_rv in c.predict_outputs_by_name(loaded, test_data).items():
            for rv, v in per_rv.items():
                arrays[f"{kind}_pred_{key}__{rv}"] = v
    save_npz(out_dir, "multi", arrays)


def make_predict(out_dir: Path, work_dir: Path) -> None:
    """Pin ``NormativeModel.predict`` (one call) of two saved models.

    Runs after ``make_blr`` and ``make_hbr``: it reads their held-out data
    and saved models from ``out_dir``.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    work_dir : Path
        Scratch directory for the model copies and their result files.
    """
    arrays: dict[str, Any] = {}
    for name, (npz, prefix, _) in c.PREDICT_CASES.items():
        golden = c.load_npz(npz, out_dir)
        data = c.to_normdata(
            "test",
            golden[f"{prefix}test_X"],
            golden[f"{prefix}test_be"],
            golden[f"{prefix}test_Y"],
        )
        model = c.copy_model(name, work_dir / "predict", golden_dir=out_dir)
        for k, v in c.full_predict_outputs(model, data).items():
            arrays[f"{name}_{k}"] = np.asarray(v)
    save_npz(out_dir, "predict", arrays)


def git_commit() -> str:
    """Return the current commit, with ``-dirty`` if the tree has changes.

    Returns
    -------
    str
        Commit hash, or ``"unknown"``.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "pcntoolkit"], cwd=REPO_ROOT
        ).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """Generate every golden file."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="output directory",
    )
    args = parser.parse_args()
    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    make_math(out_dir)
    make_transforms(out_dir)
    make_normdata_eval(out_dir)
    make_shash_chunks(out_dir)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/var/tmp")) as tmp:
        make_blr(out_dir, Path(tmp))
        rhat = make_hbr(out_dir, Path(tmp))
        make_multi(out_dir, Path(tmp))
        make_predict(out_dir, Path(tmp))

    provenance = {
        "commit": git_commit(),
        "pcntoolkit": pcntoolkit.__version__
        if hasattr(pcntoolkit, "__version__")
        else "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pymc": pm.__version__,
        "pytensor": pytensor.__version__,
        "arviz": az.__version__,
        "xarray": xr.__version__,
        "platform": platform.platform(),
        "seed": c.SEED,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "hbr_max_rhat": rhat,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
