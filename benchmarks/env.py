"""Record the software and hardware environment of a benchmark run.

This module only reads the environment. It never changes thread limits:
set ``OMP_NUM_THREADS`` and similar variables yourself before you start a
benchmark.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "PYTENSOR_FLAGS",
)

PACKAGES = (
    "pcntoolkit",
    "numpy",
    "scipy",
    "xarray",
    "pandas",
    "pymc",
    "pytensor",
    "nutpie",
    "numba",
    "arviz",
    "dask",
)

REPO_DIR = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """Run a git command in the repository and return its stdout.

    Parameters
    ----------
    *args : str
        Arguments to ``git``.

    Returns
    -------
    str | None
        Stripped stdout, or None if git failed or is missing.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    """Return the git commit and dirty flags of the repository.

    Returns
    -------
    dict[str, Any]
        ``commit``, ``branch``, ``dirty`` (any change to tracked files) and
        ``dirty_pcntoolkit`` (any change, tracked or not, under
        ``pcntoolkit/``).
    """
    commit = _git("rev-parse", "HEAD")
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    lib = _git("status", "--porcelain", "--", "pcntoolkit")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if tracked is None else bool(tracked),
        "dirty_pcntoolkit": None if lib is None else bool(lib),
    }


def cpu_model() -> str:
    """Return a human-readable CPU model name.

    Returns
    -------
    str
        CPU model, or ``platform.processor()`` if it cannot be found.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor()


def blas_info() -> Any:
    """Return numpy's build and BLAS/LAPACK configuration.

    Returns
    -------
    Any
        ``numpy.show_config(mode="dicts")`` if supported, else the printed
        text.
    """
    import numpy as np

    try:
        return np.show_config(mode="dicts")
    except TypeError:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        return buf.getvalue()


def threadpool_info() -> Any:
    """Return threadpoolctl's view of the loaded thread pools.

    Returns
    -------
    Any
        List of thread pool descriptions, or None if threadpoolctl is not
        installed. This only reads; it never sets limits.
    """
    try:
        from threadpoolctl import threadpool_info as _info
    except ImportError:
        return None
    return _info()


def collect_env() -> dict[str, Any]:
    """Collect everything needed to compare two benchmark runs.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable description of the environment.
    """
    versions: dict[str, str | None] = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git": git_info(),
        "versions": versions,
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": affinity,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "threadpools": threadpool_info(),
        "numpy_config": blas_info(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(collect_env(), indent=2, default=str))
