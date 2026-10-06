"""Tests for result part files of Runner jobs and their merge (#575).

In a Runner job, ``NormData.save_results`` writes part files and does not
read or rewrite the shared CSV files. ``merge_result_parts`` (called by
``load_results``) joins the parts into the same CSV files.
"""

from __future__ import annotations

import os
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pcntoolkit.dataio import norm_data as nd
from pcntoolkit.dataio.norm_data import NormData, merge_result_parts, result_parts
from pcntoolkit.util.runner import load_and_execute

N, R = 12, 6
KINDS = list(nd.RESULT_KEYS)


def _data(name: str = "test", seed: int = 0) -> NormData:
    """NormData with Z, centiles, logp and statistics for R response variables."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 1))
    Y = rng.normal(size=(N, R))
    be = np.repeat([["a"]], N, axis=0)
    data = NormData.from_ndarrays(name, X, Y, be)
    obs, rv = data.observations.values, data.response_vars.values
    dims = ("observations", "response_vars")
    coords = {"observations": obs, "response_vars": rv}
    data["Z"] = xr.DataArray(rng.normal(size=(N, R)), dims=dims, coords=coords)
    data["logp"] = xr.DataArray(rng.normal(size=(N, R)), dims=dims, coords=coords)
    data["centiles"] = xr.DataArray(
        rng.normal(size=(3, N, R)),
        dims=("centile", *dims),
        coords={"centile": [0.05, 0.5, 0.95], **coords},
    )
    data["statistics"] = xr.DataArray(
        rng.normal(size=(R, 2)),
        dims=("response_vars", "statistic"),
        coords={"response_vars": rv, "statistic": ["Rho", "RMSE"]},
    )
    return data


def _write_in_jobs(data: NormData, save_dir: Path, n_jobs: int) -> None:
    for i, chunk in enumerate(data.chunk(n_jobs)):
        with result_parts(f"task_2026-10-06_12:00:00_job_{i}"):
            chunk.save_results(str(save_dir))


def _read(save_dir: Path, kind: str, name: str = "test") -> pd.DataFrame:
    return nd._read_result(str(save_dir / f"{kind}_{name}.csv"), kind)


def test_resultParts_should_writeOnlyPartFiles_when_inRunnerJob(tmp_path: Path) -> None:
    """
    Arrange: results of 3 jobs.
    Act: save_results inside result_parts.
    Assert: no shared CSV file; one part per job and kind in parts/.
    """
    _write_in_jobs(_data(), tmp_path, 3)
    assert not list(tmp_path.glob("*.csv"))
    parts = sorted(p.name for p in (tmp_path / "parts").glob("*.csv"))
    assert len(parts) == 3 * len(KINDS)
    assert all(":" not in p for p in parts)


def test_mergeResultParts_should_giveSameResultsAsSharedFiles_when_jobsDone(
    tmp_path: Path,
) -> None:
    """
    Arrange: the same results written by 3 jobs into the shared files (old
        way) and as part files (new way).
    Act: merge_result_parts on the parts.
    Assert: same rows and columns; the values of the merged file are exactly
        the values in memory, and the old files differ by at most 1e-14
        (their repeated read and rewrite rounds the last digit).
    """
    data = _data()
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    for chunk in data.chunk(3):
        chunk.save_results(str(old_dir))
    _write_in_jobs(data, new_dir, 3)
    merge_result_parts(str(new_dir))
    assert not (new_dir / "parts").exists()
    for kind in KINDS:
        old, new = _read(old_dir, kind), _read(new_dir, kind)
        assert sorted(old.columns) == sorted(new.columns), kind
        old = old.loc[new.index, new.columns]
        num = [c for c in new.columns if c != "subject_ids"]
        np.testing.assert_allclose(new[num], old[num], rtol=0, atol=1e-14, err_msg=kind)
    z = _read(new_dir, "Z")
    expected = data.Z.to_pandas()
    expected.index = expected.index.astype(str)
    for rv in data.response_vars.values:
        np.testing.assert_array_equal(
            z[rv].to_numpy(), expected.loc[z.index, rv].to_numpy()
        )


def test_mergeResultParts_should_sortColumnsAndRows_when_merged(tmp_path: Path) -> None:
    """
    Arrange: part files of 3 jobs.
    Act: merge_result_parts.
    Assert: subject_ids first, then response variables sorted; rows sorted
        by observation (and centile).
    """
    _write_in_jobs(_data(), tmp_path, 3)
    merge_result_parts(str(tmp_path), "test")
    for kind in KINDS:
        header = (tmp_path / f"{kind}_test.csv").read_text().splitlines()[0].split(",")
        keys = nd.RESULT_KEYS[kind]
        assert header[: len(keys)] == keys
        rest = header[len(keys) :]
        first = ["subject_ids"] if kind != "statistics" else []
        assert rest == first + sorted(c for c in rest if c not in first), kind
    centiles = pd.read_csv(tmp_path / "centiles_test.csv")
    order = list(zip(centiles["observations"], centiles["centile"], strict=True))
    assert order == sorted(order)


def test_mergeResultParts_should_keepOldColumnsAndOverride_when_fileExists(
    tmp_path: Path,
) -> None:
    """
    Arrange: an existing Z file with rv0..rv5 (seed 0), then parts of new
        results (seed 1) for rv0..rv2 only.
    Act: merge_result_parts.
    Assert: rv0..rv2 have the new values; rv3..rv5 keep the old values.
    """
    _data(seed=0).save_results(str(tmp_path))
    new = _data(seed=1)
    _write_in_jobs(new.isel(response_vars=slice(0, 3)), tmp_path, 1)
    old_z = _read(tmp_path, "Z")
    merge_result_parts(str(tmp_path))
    z = _read(tmp_path, "Z")
    expected = new.Z.to_pandas()
    expected.index = expected.index.astype(str)
    rvs = list(new.response_vars.values)
    for rv in rvs[:3]:
        np.testing.assert_array_equal(
            z[rv].to_numpy(), expected.loc[z.index, rv].to_numpy()
        )
    for rv in rvs[3:]:
        np.testing.assert_array_equal(
            z[rv].to_numpy(), old_z.loc[z.index, rv].to_numpy()
        )


def test_mergeResultParts_should_changeNothing_when_runAgainOrNoParts(
    tmp_path: Path,
) -> None:
    """
    Arrange: merged results; and a folder without parts.
    Act: merge_result_parts again; and on the empty folder.
    Assert: the files do not change, and nothing fails.
    """
    _write_in_jobs(_data(), tmp_path, 2)
    merge_result_parts(str(tmp_path))
    before = {k: (tmp_path / f"{k}_test.csv").read_bytes() for k in KINDS}
    merge_result_parts(str(tmp_path))
    assert before == {k: (tmp_path / f"{k}_test.csv").read_bytes() for k in KINDS}
    merge_result_parts(str(tmp_path / "missing"))


def test_loadResults_should_mergeParts_when_partsExist(tmp_path: Path) -> None:
    """
    Arrange: part files of 2 jobs for "test" and of 1 job for "other".
    Act: load_results on a NormData named "test".
    Assert: the results are loaded (rtol 1e-14); only the "test" parts were
        merged.
    """
    data = _data()
    _write_in_jobs(data, tmp_path, 2)
    _write_in_jobs(_data(name="other"), tmp_path, 1)
    target = _data()
    for var in ("Z", "centiles", "logp", "statistics"):
        del target[var]
    target.load_results(str(tmp_path))
    # load_zscores reads with pandas' default float parser, which can change
    # the last digit; the merged file itself is exact (see the test above).
    np.testing.assert_allclose(
        target.Z.sel(response_vars=data.response_vars).values,
        data.Z.values,
        rtol=1e-14,
    )
    left = sorted(p.name for p in (tmp_path / "parts").glob("*.csv"))
    assert left and all(p.split("_", 1)[1].startswith("other__part__") for p in left)


def _job_fn(data: NormData, save_dir: str) -> None:
    data.save_results(save_dir)


def test_loadAndExecute_should_writePartFiles_when_runAsJob(tmp_path: Path) -> None:
    """
    Arrange: a pickled callable and data, as Runner.save_callable_and_data
        writes them for a scheduler job.
    Act: load_and_execute (the entry point of each job).
    Assert: the job wrote part files tagged with its job name, not the
        shared files; after the job, results are written directly again.
    """
    job = "task_2026-10-06_12:00:00_job_3"
    callable_path = tmp_path / f"python_callable_{job}.pkl"
    data_path = tmp_path / f"data_{job}.pkl"
    results = tmp_path / "results"
    results.mkdir()
    callable_path.write_bytes(cloudpickle.dumps(_job_fn))
    data_path.write_bytes(cloudpickle.dumps((_data(), str(results))))
    load_and_execute([str(callable_path), str(data_path), "0"])
    parts = sorted(os.listdir(results / "parts"))
    assert parts and all("__part__task_2026-10-06_12_00_00_job_3" in p for p in parts)
    assert not list(results.glob("*.csv"))
    assert nd._RESULT_PART_TAG is None


@pytest.mark.parametrize("name", ["a.b", "x[1]", "with space"])
def test_mergeResultParts_should_handleSpecialCharacters_when_inDataName(
    tmp_path: Path, name: str
) -> None:
    """
    Arrange: part files for a data name with glob or dot characters.
    Act: merge_result_parts for that name.
    Assert: the merged file exists and no part is left.
    """
    _write_in_jobs(_data(name=name), tmp_path, 2)
    merge_result_parts(str(tmp_path), name)
    assert (tmp_path / f"Z_{name}.csv").is_file()
    assert not (tmp_path / "parts").exists()


def test_mergeResultParts_should_sortRows_when_existingFileHasOtherRows(
    tmp_path: Path,
) -> None:
    """
    Arrange: an existing result file for observations 6-11 only, and parts
        for all observations 0-11.
    Act: merge_result_parts.
    Assert: rows sorted by observation (and centile) in every file.
    """
    data = _data()
    data.isel(observations=slice(6, N)).save_results(str(tmp_path))
    _write_in_jobs(data, tmp_path, 2)
    merge_result_parts(str(tmp_path))
    for kind in ("Z", "centiles", "logp"):
        df = pd.read_csv(tmp_path / f"{kind}_test.csv")
        keys = [tuple(r) for r in df[nd.RESULT_KEYS[kind]].to_numpy()]
        assert keys == sorted(keys), kind
        assert len(set(df["observations"])) == N, kind


def test_mergeResultParts_should_keepStatisticsExact_when_merged(
    tmp_path: Path,
) -> None:
    """
    Arrange: part files of 3 jobs.
    Act: merge_result_parts.
    Assert: the statistics in the merged file equal the values in memory.
    """
    data = _data()
    _write_in_jobs(data, tmp_path, 3)
    merge_result_parts(str(tmp_path))
    stats = _read(tmp_path, "statistics")
    expected = data.statistics.to_pandas().T
    for rv in data.response_vars.values:
        np.testing.assert_array_equal(
            stats.loc[expected.index, rv].to_numpy(), expected[rv].to_numpy()
        )


def test_runnerMergeResultParts_should_mergeFoldFolders_when_crossValidating(
    tmp_path: Path,
) -> None:
    """
    Arrange: part files in <save_dir>/results and in two fold result folders.
    Act: Runner.merge_result_parts (called when the Runner observes the jobs).
    Assert: every folder has merged CSV files and no parts left.
    """
    from pcntoolkit.util.runner import Runner

    dirs = [tmp_path / "results"] + [
        tmp_path / "folds" / f"fold_{i}" / "results" for i in range(2)
    ]
    for d in dirs:
        _write_in_jobs(_data(), d, 2)
    runner = Runner.__new__(Runner)
    runner.save_dir = str(tmp_path)
    runner.merge_result_parts()
    for d in dirs:
        assert (d / "Z_test.csv").is_file()
        assert not (d / "parts").exists()
