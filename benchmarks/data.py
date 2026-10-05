"""Datasets for the benchmarks.

Two sources:

- Real data: the FCON1000 cortical thickness set, downloaded once with
  ``pcntoolkit.load_fcon1000`` into ``benchmarks/data/`` (git-ignored).
- Synthetic data: a seeded generator with a known heteroskedastic true model.
  The same seed always gives the same numbers, so two code versions see the
  same input.

The synthetic generator does not use ``test/fixtures/data_fixtures.py``,
because those fixtures are not seeded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pcntoolkit import NormData, load_fcon1000

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
MODELS_DIR = BENCH_DIR / "models"
RESULTS_DIR = BENCH_DIR / "results"


def load_fcon(n_response_vars: int | None = None) -> NormData:
    """Load FCON1000 and keep only the first response variables.

    Parameters
    ----------
    n_response_vars : int | None
        Number of response variables to keep, in file order. ``None`` keeps
        all of them.

    Returns
    -------
    NormData
        FCON1000 data with covariate ``age`` and batch effects ``sex`` and
        ``site``.
    """
    data = load_fcon1000(save_path=str(DATA_DIR))
    if n_response_vars is not None:
        data = select_response_vars(data, n_response_vars)
    return data


def select_response_vars(data: NormData, n_response_vars: int) -> NormData:
    """Return a copy of ``data`` that keeps the first ``n_response_vars``.

    Parameters
    ----------
    data : NormData
        Input data.
    n_response_vars : int
        Number of response variables to keep.

    Returns
    -------
    NormData
        Copy of the data with fewer response variables.

    Raises
    ------
    ValueError
        If ``n_response_vars`` is larger than the number available.
    """
    names = list(data.response_vars.values)
    if n_response_vars > len(names):
        raise ValueError(
            f"Requested {n_response_vars} response vars, data has {len(names)}."
        )
    subset = data.sel(response_vars=names[:n_response_vars]).copy(deep=True)
    subset.attrs = dict(data.attrs)
    return subset


def fcon_train_test(
    n_response_vars: int, seed: int = 0, test_frac: float = 0.3
) -> tuple[NormData, NormData]:
    """Split FCON1000 into seeded train and test sets.

    The split is stratified on ``site`` so every site is in both sets.

    Parameters
    ----------
    n_response_vars : int
        Number of response variables to keep.
    seed : int
        Seed for the split.
    test_frac : float
        Fraction of each site that goes to the test set.

    Returns
    -------
    tuple[NormData, NormData]
        Train and test data.
    """
    data = load_fcon()
    df = data.to_dataframe()
    rng = np.random.default_rng(seed)
    site_col = ("batch_effects", "site")
    is_test = np.zeros(len(df), dtype=bool)
    for site in np.unique(df[site_col].to_numpy()):
        idx = np.flatnonzero(df[site_col].to_numpy() == site)
        n_test = max(1, int(round(test_frac * idx.size)))
        is_test[rng.choice(idx, size=n_test, replace=False)] = True
    names = [str(v) for v in data.response_vars.values[:n_response_vars]]

    def build(mask: np.ndarray, name: str) -> NormData:
        part = df[mask]
        flat = pd.DataFrame(
            {
                "age": part[("X", "age")].to_numpy(),
                "sex": part[("batch_effects", "sex")].to_numpy(),
                "site": part[("batch_effects", "site")].to_numpy(),
                **{rv: part[("Y", rv)].to_numpy() for rv in names},
            }
        )
        return NormData.from_dataframe(
            name,
            flat,
            covariates=["age"],
            batch_effects=["sex", "site"],
            response_vars=names,
        )

    return build(~is_test, "fcon_train"), build(is_test, "fcon_test")


def make_synthetic(
    n_subjects: int,
    n_response_vars: int,
    n_batch_levels: int = 2,
    n_batch_dims: int = 1,
    seed: int = 0,
    split: int = 0,
    name: str = "synthetic",
    return_truth: bool = False,
) -> NormData | tuple[NormData, dict[str, np.ndarray]]:
    """Generate seeded synthetic data with a known heteroskedastic model.

    For subject ``i`` and response variable ``r``, with ``x`` the age
    rescaled to [0, 1]:

    - mean: ``a_r + b_r * x + c_r * sin(2 pi x) + sum_d offset[r, d, level_d]``
    - SD: ``exp(s0_r + s1_r * x)`` (the spread changes with age)
    - ``Y = mean + SD * e`` with ``e`` drawn from a standard normal.

    Batch levels are given round-robin and then shuffled, so every level is
    present when ``n_subjects >= n_batch_levels``.

    Parameters
    ----------
    n_subjects : int
        Number of observations.
    n_response_vars : int
        Number of response variables.
    n_batch_levels : int
        Number of levels in each batch-effect dimension (e.g. sites).
    n_batch_dims : int
        Number of batch-effect dimensions.
    seed : int
        Seed of the true model (coefficients, batch offsets). Same seed and
        sizes give the same model.
    split : int
        Seed of the observations drawn from that model. Use different values
        for a train and a test set from the same model.
    name : str
        Name of the NormData object.
    return_truth : bool
        If True, also return the true mean and SD per observation.

    Returns
    -------
    NormData | tuple[NormData, dict[str, np.ndarray]]
        The data, and if ``return_truth`` is set a dict with ``mean`` and
        ``sd`` arrays of shape (n_subjects, n_response_vars).
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(2.5, 0.3, n_response_vars)
    b = rng.normal(-0.5, 0.2, n_response_vars)
    c = rng.normal(0.0, 0.1, n_response_vars)
    s0 = rng.normal(-2.0, 0.2, n_response_vars)
    s1 = rng.normal(0.8, 0.2, n_response_vars)
    offsets = rng.normal(0.0, 0.1, (n_response_vars, n_batch_dims, n_batch_levels))

    # A second stream for the observations, so train and test share one model.
    rng = np.random.default_rng([seed, split])
    age = rng.uniform(10.0, 90.0, n_subjects)
    x = (age - 10.0) / 80.0
    levels = np.empty((n_subjects, n_batch_dims), dtype=np.int64)
    for d in range(n_batch_dims):
        levels[:, d] = rng.permutation(np.arange(n_subjects) % n_batch_levels)

    mean = (
        a[None, :]
        + b[None, :] * x[:, None]
        + c[None, :] * np.sin(2 * np.pi * x)[:, None]
    )
    for d in range(n_batch_dims):
        mean += offsets[:, d, :][:, levels[:, d]].T
    sd = np.exp(s0[None, :] + s1[None, :] * x[:, None])
    y = mean + sd * rng.standard_normal((n_subjects, n_response_vars))

    be_names = [f"be{d}" for d in range(n_batch_dims)]
    rv_names = [f"rv{r}" for r in range(n_response_vars)]
    frame: dict[str, np.ndarray] = {
        "subject_ids": np.array([f"sub{i}" for i in range(n_subjects)]),
        "age": age,
    }
    for d, be in enumerate(be_names):
        # Zero-padded labels keep the sorted order equal to the level order.
        frame[be] = np.array([f"L{v:03d}" for v in levels[:, d]])
    for r, rv in enumerate(rv_names):
        frame[rv] = y[:, r]
    data = NormData.from_dataframe(
        name,
        pd.DataFrame(frame),
        covariates=["age"],
        batch_effects=be_names,
        response_vars=rv_names,
        subject_ids="subject_ids",
    )
    if return_truth:
        return data, {"mean": mean, "sd": sd}
    return data


def make_dataset(
    source: str,
    n_subjects: int,
    n_response_vars: int,
    n_batch_levels: int = 2,
    n_batch_dims: int = 1,
    seed: int = 0,
) -> tuple[NormData, NormData]:
    """Return a train and test set for one benchmark case.

    Parameters
    ----------
    source : str
        ``"synthetic"`` or ``"fcon"``. For ``"fcon"`` the size and batch
        arguments are ignored (FCON1000 has a fixed size and batch effects);
        only ``n_response_vars`` is used.
    n_subjects : int
        Number of observations in each of the train and test sets.
    n_response_vars : int
        Number of response variables.
    n_batch_levels : int
        Levels per batch-effect dimension (synthetic only).
    n_batch_dims : int
        Number of batch-effect dimensions (synthetic only).
    seed : int
        Seed of the true model (synthetic) or of the split (fcon).

    Returns
    -------
    tuple[NormData, NormData]
        Train and test data.

    Raises
    ------
    ValueError
        If ``source`` is not known.
    """
    if source == "synthetic":
        train = make_synthetic(
            n_subjects, n_response_vars, n_batch_levels, n_batch_dims, seed, 0, "train"
        )
        test = make_synthetic(
            n_subjects, n_response_vars, n_batch_levels, n_batch_dims, seed, 1, "test"
        )
        return train, test  # type: ignore[return-value]
    if source == "fcon":
        return fcon_train_test(n_response_vars, seed=seed)
    raise ValueError(f"Unknown data source {source!r}")
