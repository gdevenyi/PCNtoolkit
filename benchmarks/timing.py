"""Small timing and reporting helpers shared by the benchmark scripts.

What is measured: wall-clock time of one call, with ``time.perf_counter``.
Warm-up calls run first and are reported separately (the first call often
includes one-off costs such as compilation or file reads). Optional
``setup`` runs before every call and is not timed.

Memory: only one number is reported, the peak resident set size of the whole
process at the end of the script (``resource.getrusage``). It is a high-water
mark over everything the script did, not per benchmark. ``tracemalloc`` is
not used, because it does not see most numpy and BLAS allocations.
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.data import RESULTS_DIR
from benchmarks.env import collect_env


@dataclass
class Timing:
    """Result of timing one callable.

    Attributes
    ----------
    name : str
        Label of the timed operation.
    times : list[float]
        Wall time in seconds of each timed call.
    warmup_times : list[float]
        Wall time in seconds of each warm-up call (not in the statistics).
    median : float
        Median of ``times``.
    min : float
        Minimum of ``times``.
    """

    name: str
    times: list[float]
    warmup_times: list[float] = field(default_factory=list)
    median: float = float("nan")
    min: float = float("nan")

    def __post_init__(self) -> None:
        if self.times:
            self.median = statistics.median(self.times)
            self.min = min(self.times)


def time_call(
    name: str,
    fn: Callable[[], Any],
    repeats: int = 3,
    warmup: int = 1,
    setup: Callable[[], Any] | None = None,
) -> Timing:
    """Time ``fn`` with warm-up and repeats.

    Parameters
    ----------
    name : str
        Label of the operation.
    fn : Callable[[], Any]
        Function to time. It must force all lazy work (e.g. call
        ``np.asarray`` on a dask result) before it returns.
    repeats : int
        Number of timed calls.
    warmup : int
        Number of untimed-for-statistics warm-up calls.
    setup : Callable[[], Any] | None
        Called before every call (warm-up and timed); not timed. Use it to
        clear caches or make a fresh model.

    Returns
    -------
    Timing
        Times of the warm-up and timed calls.
    """
    warm: list[float] = []
    times: list[float] = []
    for i in range(warmup + repeats):
        if setup is not None:
            setup()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        (warm if i < warmup else times).append(dt)
    return Timing(name=name, times=times, warmup_times=warm)


def peak_rss_mb() -> float | None:
    """Return the peak resident memory of this process in MiB.

    Returns
    -------
    float | None
        High-water mark over the whole process lifetime, or None if the
        ``resource`` module is not available (Windows).
    """
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def write_results(
    script: str,
    params: dict[str, Any],
    cases: list[dict[str, Any]],
    out: str | None = None,
) -> Path:
    """Write a benchmark result file.

    Parameters
    ----------
    script : str
        Name of the benchmark script.
    params : dict[str, Any]
        Command-line parameters of the run.
    cases : list[dict[str, Any]]
        One dict per benchmark case. Each has a ``case`` dict (the sizes and
        options) and a ``timings`` list of :class:`Timing` (or dicts).
    out : str | None
        Output path. Default is
        ``benchmarks/results/<script>_<size>_<timestamp>.json``.

    Returns
    -------
    Path
        Path of the written file.
    """
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = (
        Path(out)
        if out
        else RESULTS_DIR / f"{script}_{params.get('size', 'custom')}_{stamp}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": script,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "env": collect_env(),
        "cases": [
            {
                **c,
                "timings": [
                    asdict(t) if isinstance(t, Timing) else t
                    for t in c.get("timings", [])
                ],
            }
            for c in cases
        ],
        "process_peak_rss_mb": peak_rss_mb(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def format_case(case: dict[str, Any]) -> str:
    """Format a case dict as a short ``key=value`` string.

    Parameters
    ----------
    case : dict[str, Any]
        Case parameters.

    Returns
    -------
    str
        Compact label.
    """
    return " ".join(f"{k}={v}" for k, v in case.items())


def print_table(cases: list[dict[str, Any]]) -> None:
    """Print one line per timing: case, operation, median, min, warm-up.

    Parameters
    ----------
    cases : list[dict[str, Any]]
        Cases as passed to :func:`write_results`.
    """
    rows = []
    for c in cases:
        label = format_case(c["case"])
        for t in c.get("timings", []):
            t = asdict(t) if isinstance(t, Timing) else t
            warm = ",".join(f"{w:.3g}" for w in t["warmup_times"]) or "-"
            rows.append(
                (
                    label,
                    t["name"],
                    f"{t['median']:.4g}",
                    f"{t['min']:.4g}",
                    str(len(t["times"])),
                    warm,
                )
            )
    header = ("case", "operation", "median_s", "min_s", "n", "warmup_s")
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths, strict=True)))
