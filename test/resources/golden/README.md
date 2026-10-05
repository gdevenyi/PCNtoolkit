# Golden values

The files in this directory pin the numeric output of the PCNtoolkit hot spots
that performance PRs will change. The tests in `test/test_golden/` recompute
each output from the stored inputs and compare it with the stored value.

## The rule

**Regenerate these files only when you change the behaviour on purpose.**

This is the opposite of the rule for `test/resources/pretrained_*`, which must
never be regenerated. Here, regeneration is allowed, but only for an
intentional behaviour change (for example, a bug fix that changes results).
A speedup PR must not regenerate them: the point of the files is to show that
the faster code gives the same numbers as the old code. If a speedup PR makes
a golden test fail, fix the code or explain in the PR why the old value was
wrong.

## How to regenerate

From the repository root:

```bash
.venv/bin/python test/resources/golden/generate_golden.py
.venv/bin/python -m pytest test/test_golden -q
```

`--out DIR` writes to another directory. Use it to check that the generator
is reproducible: the `.npz` and `.json` files are byte-identical between runs,
and the `idata.nc` files differ only in their creation time stamp (the
posterior values are identical). Run the generator twice and compare the
two outputs: one first run on a cold PyTensor compile cache gave a ZINB
posterior that differed by about 6e-8 from all later runs. The cause is not
known.

Then update the provenance table below from `provenance.json`.

The generator records `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `SLURM_CPUS_PER_TASK` in `provenance.json`. It never
sets them.

## Tolerances

| Class | Applies to | Tolerance |
|---|---|---|
| DETERMINISTIC | Fixed inputs, no optimiser, no sampling: shash, likelihood maps, warps, scalers, basis functions, NormData scaling, Evaluator, BLR at fixed hyperparameters, BLR and HBR predictions from a saved model | `rtol=1e-10`; values that can be close to 0 also get `atol=1e-12` |
| OPTIMISER | BLR re-fit from the stored training data | converged negative log-likelihood `rtol=1e-8`; Z-scores `atol=1e-6` |

The tests never re-fit an HBR model (MCMC). They only predict from the saved
posterior. Prediction from a fixed posterior only evaluates deterministic
functions of it, so the DETERMINISTIC class applies.

Example: a B-spline basis column that is 0 outside its support stays 0 in a
correct speedup, but may come back as 3e-17. `rtol` alone would fail on that,
so `atol=1e-12` covers it. A value of 2.5 must still match to about 2.5e-10.

## Contents

| File | Content |
|---|---|
| `math.npz` | shash `S`, `S_inv`, `K`, `P`, `m1m2`; numpy `forward`/`backward`/`yhat` of the Normal, SHASHb, SHASHo, SHASHo2, Beta and ZINB likelihoods; warps (Log, Affine, BoxCox, SinhArcsinh, Compose) `f`/`invf`/`df` |
| `transforms.npz` | scalers (standardize, minmax, robminmax, id, with and without clipping); B-spline, polynomial and linear basis functions |
| `normdata_eval.npz` | `NormData.scale_forward`/`scale_backward`; all Evaluator statistics plus BIC; `iter_batch_combinations`; `HBR.extract_and_reshape` |
| `blr.npz` | BLR inputs; `post`, `loglik`, `penalized_loglik`, `dloglik`, `ys_s2`, `forward`, `backward`, `elemwise_logp` at fixed hyperparameters; fitted hyperparameters and negative log-likelihood; predictions of the saved models; `sample_batch_effects` and `sample_covariates` with a fixed global seed |
| `blr_plain/`, `blr_warp/`, `blr_hetero/` | Saved BLR models (l-bfgs-b; no warp, SinhArcsinh warp, heteroskedastic noise) |
| `hbr.npz` | HBR inputs and predictions (`compute_zscores`, `compute_centiles`, `compute_logp`, `compute_yhat`) of the saved models |
| `hbr_Normal/`, `hbr_SHASHb/`, `hbr_beta/`, `hbr_ZINB/` | Saved HBR models (pymc sampler, 2 chains, 20 tuning and 20 draws, fixed `random_seed`) |
| `provenance.json` | Commit, package versions, platform, seed and thread variables of the last run |

All inputs come from `np.random.default_rng` with seeds derived from one
master seed. The shared code is in `test/test_golden/_common.py`.

The HBR chains are deliberately too short to converge (R-hat up to 1.45).
This does not matter: the tests check that prediction from a fixed posterior
does not change, not that the posterior is correct.

## Known gaps

- **SHASHo and SHASHo2 HBR models.** `SHASHoLikelihood` and
  `SHASHo2Likelihood` are abstract classes (they do not implement
  `compile_params`, `transfer`, `_update_data` and `yhat`), and
  `Likelihood.from_dict` does not know them. No HBR model can use them. Only
  their numpy `forward`/`backward` maps are pinned, in `math.npz`.
- **ZINB Z-scores.** `ZeroInflatedNegativeBinomialLikelihood.forward` draws
  random numbers, and `HBR.forward` cannot pass it a seeded generator. ZINB
  Z-scores from a saved model are therefore checked only for shape and
  finiteness. The numpy `forward` with an explicit seeded generator is pinned
  in `math.npz`.

## Provenance

| Item | Value |
|---|---|
| Commit | `fdbdbbbc4cccef831bf61c7c0736c778eda80d18` (branch `gdevenyi/perf-benchmarks`, no changes under `pcntoolkit/`) |
| PCNtoolkit | 1.4.0 |
| Python | 3.13.12 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| PyMC / PyTensor / ArviZ / xarray | 6.3.2 / 3.3.2 / 1.3.0 / 2026.7.0 |
| Platform | Linux x86_64 (glibc 2.44) |
| Thread variables | none set |
| Total size | about 0.5 MB |
