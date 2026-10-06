"""Tests for the worker-count rules in ``pcntoolkit.util.parallel``."""

from __future__ import annotations

import os
import warnings

import pytest

from pcntoolkit.util import parallel
from pcntoolkit.util.output import Output
from pcntoolkit.util.parallel import allocated_cpus, map_tasks, resolve_n_jobs


@pytest.fixture
def eight_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the process can run on 8 CPUs, with no allocation variables."""
    monkeypatch.setattr(parallel, "cpu_count", lambda only_physical_cores=False: 8)
    for name in parallel.ALLOCATION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _square(x: int) -> int:
    return x * x


def _worker_pid(_: int) -> int:
    return os.getpid()


def _show_messages(_: int) -> bool:
    return Output._show_messages


def _warn(x: int) -> int:
    warnings.warn(f"worker warning {x}", UserWarning, stacklevel=1)
    return x


def test_001_allocatedCpus_should_useCpuCount_when_noEnvVars(eight_cpus: None) -> None:
    assert allocated_cpus() == 8


@pytest.mark.parametrize("name", parallel.ALLOCATION_ENV_VARS)
def test_002_allocatedCpus_should_useEnvVar_when_smallerThanCpuCount(
    eight_cpus: None, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "3")
    assert allocated_cpus() == 3


def test_003_allocatedCpus_should_takeSmallest_when_bothEnvVarsSet(
    eight_cpus: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert allocated_cpus() == 2


@pytest.mark.parametrize("name", ["PBS_NUM_PPN", "NCPUS"])
def test_011_allocatedCpus_should_usePbsVar_when_set(
    eight_cpus: None, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "5")
    assert allocated_cpus() == 5


def test_012_allocatedCpus_should_useFirstValue_when_ompListForm(
    eight_cpus: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "2,1")
    assert allocated_cpus() == 2


@pytest.mark.parametrize("value", ["", "abc", "0", "-2", "16"])
def test_004_allocatedCpus_should_ignoreEnvVar_when_invalidOrLarger(
    eight_cpus: None, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", value)
    assert allocated_cpus() == 8


@pytest.mark.parametrize(
    ("n_jobs", "expected"), [(1, 1), (4, 4), (8, 8), (-1, 8), (-2, 7), (-20, 1)]
)
def test_005_resolveNJobs_should_followJoblibRules_when_withinAllocation(
    eight_cpus: None, n_jobs: int, expected: int
) -> None:
    assert resolve_n_jobs(n_jobs) == expected


def test_006_resolveNJobs_should_capAndWarn_when_aboveAllocation(
    eight_cpus: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.warns(UserWarning, match="n_jobs=6"):
        assert resolve_n_jobs(6) == 2


def test_016_resolveNJobs_should_acceptNumpyInteger_when_given(
    eight_cpus: None,
) -> None:
    import numpy as np

    assert resolve_n_jobs(np.int64(3)) == 3


@pytest.mark.parametrize("n_jobs", [0, 1.5, True, "2"])
def test_007_resolveNJobs_should_raise_when_notNonZeroInteger(n_jobs: object) -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        resolve_n_jobs(n_jobs)  # type: ignore[arg-type]


def test_008_mapTasks_should_runInThisProcess_when_oneWorker() -> None:
    assert list(map_tasks(_worker_pid, [(0,), (1,)], 2, 1)) == [os.getpid()] * 2


def test_009_mapTasks_should_runInThisProcess_when_oneTask() -> None:
    assert list(map_tasks(_worker_pid, [(0,)], 1, 4)) == [os.getpid()]


def test_010_mapTasks_should_keepTaskOrder_when_twoWorkers() -> None:
    tasks = [(i,) for i in range(7)]
    assert list(map_tasks(_square, tasks, 7, 2)) == [i * i for i in range(7)]
    assert os.getpid() not in list(map_tasks(_worker_pid, tasks, 7, 2))


def test_013_mapTasks_should_buildOneTaskAtATime_when_oneWorker() -> None:
    built: list[int] = []

    def tasks():
        for i in range(3):
            built.append(i)
            yield (i,)

    results = map_tasks(_square, tasks(), 3, 1)
    assert built == []
    assert next(results) == 0
    assert built == [0]


def test_014_mapTasks_should_copyOutputSettings_when_twoWorkers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Output, "_show_messages", False)
    assert list(map_tasks(_show_messages, [(0,), (1,)], 2, 2)) == [False, False]


def test_015_mapTasks_should_reraiseWorkerWarnings_when_twoWorkers() -> None:
    with pytest.warns(UserWarning) as record:
        assert list(map_tasks(_warn, [(0,), (1,)], 2, 2)) == [0, 1]
    messages = {str(w.message) for w in record}
    assert {"worker warning 0", "worker warning 1"} <= messages
