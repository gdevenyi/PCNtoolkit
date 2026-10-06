# Benchmarks

These scripts measure the speed of PCNtoolkit and check that a change does
not change the results. Use them in every performance PR:

1. Run a benchmark on `dev` and on your branch.
2. Show the speed-up with `python -m benchmarks.compare timing OLD.json NEW.json`.
3. Show that the results stay within the tolerance class of the code you
   changed (see [Tolerance classes](#tolerance-classes)).

CI does not run the benchmarks. They are for local use only.

## Set up

Use the development environment of the repository:

- `make venv` (added by a pending PR), or
- conda, as described in
  [CONTRIBUTING](https://pcntoolkit.readthedocs.io/en/stable/pages/contributing.html).

Run every command from the repository root, as a module (`python -m
benchmarks.<script>`). The scripts write to these folders, which git
ignores:

| Folder                | Contents                                              |
|-----------------------|-------------------------------------------------------|
| `benchmarks/data/`    | Cached FCON1000 download (`--data fcon`)              |
| `benchmarks/models/`  | Saved HBR models, reused between runs                 |
| `benchmarks/results/` | One JSON file per run                                 |

## Threads

The benchmarks never change thread settings. Set them yourself before you
start, and use the same settings for the "before" and "after" runs:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
```

Each JSON file records these variables, `SLURM_CPUS_PER_TASK`, the CPU
model and count, the BLAS library and the `threadpoolctl` view of the thread
pools. PCNtoolkit itself must never set global thread limits.

## Scripts

Each script takes `--size {smoke,small,medium,large}`. `smoke` takes less
than 2 minutes on a laptop and only checks that the script works; its times
are too short and noisy to compare (one smoke operation differed 2x between
two runs). Use `small` or larger, and the same machine, for real numbers. Add `--scratch DIR` to put temporary model files
in `DIR` (default: the system temporary folder). Use `--help` for all
options.

| Command                                         | What it measures |
|-------------------------------------------------|------------------|
| `python -m benchmarks.bench_blr --size small`   | BLR normative model: fit, load, each predict step, `predict`, `Evaluator.evaluate` |
| `python -m benchmarks.bench_hbr --size small`   | HBR: predict steps from a saved model, and fit time |
| `python -m benchmarks.bench_components --size small` | Single hot-spot functions (SHASH, likelihoods, BLR internals, warps, scalers, B-spline, NormData scaling, Evaluator metrics, ...) |
| `python -m benchmarks.check_hbr_accuracy`       | Not a timing: checks two HBR fits agree within MCMC noise |

### Data

`--data synthetic` (default) uses a seeded generator in `data.py`:
`make_synthetic(n_subjects, n_response_vars, n_batch_levels, n_batch_dims,
seed, split)`. The same seed always gives the same numbers. The true model is
known: the mean is a smooth function of age plus a batch offset, and the SD
grows with age (heteroskedastic). Train and test sets come from the same
model (`split=0` and `split=1`).

`--data fcon` uses the real FCON1000 data (N = 1078, batch effects sex and
site with 23 sites), split 70/30 per site. It ignores the N and batch-level
grid and keeps only the first R response variables.

### Size presets

N is the size of each of the train and test sets, R is the number of
response variables, "levels" is the number of batch levels (e.g. sites).

| Script     | smoke | small | medium | large |
|------------|-------|-------|--------|-------|
| bench_blr  | N=500, R=2, levels=2 | N=1000; R 1, 10; levels 2, 20 | N 1000, 10000; R 1, 10; levels 2, 20 | N 1000, 10000, 50000; R 1, 10, 100; levels 2, 20 (hetero and warp only) |
| bench_hbr predict | N=200, 100 samples | N=1000; levels 2, 20; 400 samples | N 1000, 10000; levels 2, 20; 2000 samples | N 1000, 10000, 50000; R 1, 10; levels 2, 20; 6000 samples (library default) |
| bench_hbr fit | N=200, tune=draws=50, 2 chains | N=1000; levels 2, 20; tune=draws=200, 2 chains | N 1000, 10000 (short), plus one default-sampling fit at N=1000 | N up to 50000 (short), plus one default-sampling fit at N=1000 |
| bench_components | N=1000, S=200 | N=5000, S=1000 | N=10000, S=2000 | N=50000, S=1000, R=100 |

Rough wall time on an 8-core laptop (smoke and small measured, the rest
estimated; heteroskedastic BLR fits with many batch levels dominate):

| Script           | smoke  | small    | medium    | large   |
|------------------|--------|----------|-----------|---------|
| bench_blr        | 10 s (fcon: 45 s) | ~30 min | hours | many hours |
| bench_hbr        | 25 s   | ~5 min   | ~1 h      | many hours |
| bench_components | 10 s   | ~2.5 h   | many hours | many hours |
| check_hbr_accuracy | 15 s | full: 25 s | -       | -       |

A heteroskedastic BLR fit with a batch effect on the noise has one
hyperparameter per batch level; at 2 x 20 levels and N=5000 one fit takes
about 2.5 minutes (one response variable). In `bench_blr`, use `--configs plain` or fewer
levels for a quick run.

The `bench_components` small time (2 h 27 min, one run; the laptop slept
once, so the true time is less) uses the default
BLAS threads. OpenBLAS thread overhead makes BLR fits 4-9x slower than with
`OMP_NUM_THREADS=1` (see #576), so set it for a faster run.

S is the number of posterior samples in the (N, S) arrays that HBR passes to
the likelihood functions. BLR configs: `plain` (B-spline mean, constant
noise), `hetero` (noise changes with age) and `warp` (hetero plus a
sinh-arcsinh warp of Y). All configs have a batch fixed effect.

### What "fit" means

`NormativeModel.fit` always ends with a full `predict` on the training data
(Z-scores, centiles, log-probabilities, Yhat). The flags `savemodel`,
`saveresults`, `saveplots` and `evaluate_model` only switch off saving,
plotting and evaluation. So the benchmarks report two numbers:

- `fit_only`: only the regression fits (one per response variable).
- `fit_total`: `NormativeModel.fit` with all four flags off, i.e.
  `fit_only` plus one predict pass.

`bench_hbr` reports only `fit_only`; it includes building the PyMC model,
compiling the sampler and sampling. The first fit in a fresh PyTensor cache
is slower.

## Reading the JSON output

Each run writes `benchmarks/results/<script>_<size>_<timestamp>.json`:

```text
params                 command-line options and the size grid
env                    git commit, dirty flags, package versions, BLAS,
                       CPU, thread variables, threadpoolctl info
cases[]                one entry per benchmark case
  case                 sizes and options (data, n, r, levels, config, ...)
  timings[]            name, times (s), warmup_times (s), median, min
  checks[]             determinism checks (name, passed, details)
process_peak_rss_mb    peak memory of the whole process (high-water mark,
                       not per case)
```

Warm-up calls are not in `median`/`min`. The first call often includes
one-off work (compilation, caches), so look at `warmup_times` too. Compare
`median` between runs; on a busy machine `min` is more stable.

Check `env.git.dirty_pcntoolkit`: it must be `false` for a "before" run on
`dev`, so you know which code you measured.

## Comparing two runs

Speed-up (old time / new time, matched by case and operation):

```bash
python -m benchmarks.compare timing benchmarks/results/OLD.json benchmarks/results/NEW.json
```

Results: run `bench_blr` or `bench_hbr` with `--save-outputs DIR` on both
branches, then

```bash
python -m benchmarks.compare outputs DIR_OLD DIR_NEW --mode deterministic
python -m benchmarks.compare outputs DIR_OLD DIR_NEW --mode optimiser   # BLR only
```

`--mode deterministic` uses `rtol=1e-10` plus `atol=1e-12`, because Z,
logp, centiles and Yhat are signed and can be close to 0.

`bench_hbr` reuses the saved models in `benchmarks/models/`, so both
branches predict from the same posterior draws; use `--mode deterministic`.
For BLR the model is refitted, so a PR that changes the fit needs
`--mode optimiser`.

For HBR fits, use `check_hbr_accuracy.py`:

```bash
git switch dev
python -m benchmarks.check_hbr_accuracy --save-reference benchmarks/models/hbr_ref
git switch my-branch
python -m benchmarks.check_hbr_accuracy --reference benchmarks/models/hbr_ref
```

Without `--save-reference`/`--reference` it fits the current code twice and
compares the two fits. The fitted problem is fixed; see
[The reference problem](#the-reference-problem-of-check_hbr_accuracy).

## Tolerance classes

The checks are in `benchmarks/compare.py` and you can import them in tests.

**DETERMINISTIC** (`compare_deterministic`). For outputs that do not depend
on an optimiser or on random sampling: BLR/HBR predictions from one saved
model, likelihood forward/backward, SHASH functions, warps, scalers, basis
functions, Evaluator metrics, NormData scaling. Rule: relative error at most
1e-10 (`rtol=1e-10`, `atol=0`). Use a tiny `atol` such as 1e-12 only where
a value can be close to 0, and write why in a comment. Example: a centile of
2.5000000001 against 2.5 has relative error 4e-11 and passes. Exception: ZINB
Z-scores use random numbers and are only reproducible with the same `rng`.

**OPTIMISER** (`compare_optimiser`). For BLR fits, because a PR may change
the optimiser internals. Rule: the converged negative log-likelihood
(`BLR.nlZ`; with l-bfgs-b this is the minimised objective, so it includes
the hyperparameter penalty) within `rtol=1e-8`, and
Z-scores within 1e-6 (absolute).

**MCMC** (`compare_mcmc`). For HBR fits. MCMC draws are random, so two fits
never agree exactly. Two terms:

- MCSE (Monte Carlo standard error): the noise in a posterior mean (or SD)
  that comes from using a finite number of draws. More draws give a smaller
  MCSE.
- R-hat: compares the chains of one fit. 1.0 means all chains sample the
  same distribution; a larger value means at least one chain is somewhere
  else (the sampler has not converged).

Rule, for every parameter:

- `|mean_a - mean_b| <= 3 * sqrt(mcse_mean_a^2 + mcse_mean_b^2)`, and the
  same for the SD with `mcse_sd`. Example: means 0.512 and 0.498 with MCSE
  0.004 each: the limit is 3 x 0.0057 = 0.017, the difference 0.014 passes.
- Z-scores within 0.05 (absolute).
- R-hat at most 1.01 in both fits.

### The reference problem of `check_hbr_accuracy`

The limits apply to one fixed problem (`--reference-model simple`, the
default): synthetic data from `make_synthetic` (N=1000 train and test, 1
response variable, 1 batch effect with 2 levels) and a Normal HBR model:

- mean: polynomial of degree 2 in age, plus a random intercept per batch
  level (centered form, `centered=True`);
- SD: softplus of a straight line in age.

This gives 8 posterior parameters. Sampling: 4 chains x 3000 draws,
tune=500 (`--size full`, about 25 s on an 8-core laptop). It still runs
the code that speed-up PRs change: the HBR fit, `forward` (Z-scores), a
covariate basis with more than one column for the mean, and the batch-effect
indexing of the random intercept.

Measured on unchanged code (30 runs in a row, in two sets of 20 and 10):
29 passed. The one failure was the SD of one mean slope at 3.5 MCSE. That is expected: 8 parameters x
2 moments = 16 tests per run, and at 3 MCSE each test fails by chance about
1 time in 370, so about 1 run in 25 fails. R-hat was at most 1.0015 and the
largest Z difference was 0.005 in all 30 runs.

The check finds real changes in fit B (3 runs each, all limits unchanged):

| Change in fit B                                   | Result     | Why |
|---------------------------------------------------|------------|-----|
| Add 0.04 to every Z-score                         | 3/3 pass   | below the 0.05 limit |
| Add 0.06 to every Z-score                         | 3/3 fail   | Z |
| Prior SD of the SD slope 2.0 -> 1.0               | 3/3 pass   | mean moved less than 3 MCSE |
| Prior SD of the SD slope 2.0 -> 0.5               | 3/3 fail   | mean moved 0.004-0.005 (6-9 MCSE, about 1%) |
| Prior SD of the SD slope 2.0 -> 0.3               | 3/3 fail   | mean moved 0.012 (20 MCSE) |
| Training Y x 1.005                                | 3/3 fail   | Z (difference 0.10) |
| Training Y x 1.02                                 | 3/3 fail   | Z (difference 0.41) |

So a change that moves one posterior mean by about 1% of its value, or
moves Z-scores by more than 0.05, fails the check. Scaling only the
training Y does not change the posterior, because the model standardizes Y
before the fit; it changes the Z-scores of the (unscaled) test set.

Why not the library default Normal model? `--reference-model hard` (B-spline
mean and SD, non-centered random intercept, 21 parameters) often does not
converge on unchanged code. In 14 runs of `--size full`, 8 failed, mostly on
R-hat: in some runs one chain did not mix with the others (R-hat above
1.01 on 18 or 19 of the 21 parameters, at most 1.06 to 1.10), and more
tuning did not fix it (`--tune 2000`: 3 of 5 runs failed). That is a problem of the model, not of
the check, and is reported separately. With the library default of 1500
draws we estimated that this model fails 40-50% of runs: 2 R-hat failures
in 5 runs, plus about 1 run in 10 for the 3-MCSE rule. That estimate was not
measured again.

Two other candidates did not pass:

- The simple model with a non-centered random intercept: 0 of 3 runs passed
  (R-hat up to 1.5, hundreds of divergences per chain). Each batch level has
  about 500 subjects, so the offsets are well determined, and the
  non-centered form then samples badly.
- The same model on FCON1000 (1 response variable, batch effect sex only):
  3 of 4 runs passed (R-hat 1.025 once). FCON1000 with site as a batch effect
  has 23 levels and so many more parameters; with 3-MCSE tests on every
  parameter it fails often by chance alone.

If a check fails, run it again, and run it on `dev`, before you blame your
change. `--max-rhat`, `--n-mcse` and `--z-atol` change the limits for an
experiment; do not change them to make a PR pass.

`check_hbr_accuracy --size smoke` uses 100 draws per chain. That is too few
for R-hat, Z and the 3-MCSE rule to be reliable, so smoke reports all three
but checks none of them. Its exit code only shows that the script runs.
