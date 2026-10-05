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

| File | Content | Size |
|---|---|---|
| `math.npz` | shash `S`, `S_inv`, `K`, `P`, `m1m2`; numpy `forward`/`backward`/`yhat` of the Normal, SHASHb, SHASHo, SHASHo2, Beta and ZINB likelihoods; warps (Log, Affine, BoxCox, SinhArcsinh, Compose) `f`/`invf`/`df` | 50 KB |
| `shash_chunks.npz` | `K` (with the `(1000, 1000)` chunks that `P` uses), `P`, `m1m2` on inputs of shape (1500, 3), (3, 1500) and (1100, 1050), which span more than one dask chunk. The inputs are not stored: `shash_chunk_inputs` builds them from a closed-form expression (no random generator). For (1100, 1050) only a summary is stored: row and column sums of squares and 1000 values at fixed positions. A 1e-9 relative change in any one chunk changes this summary by more than the tolerance | 375 KB |
| `transforms.npz` | scalers (standardize, minmax, robminmax, id, with and without clipping); B-spline, polynomial and linear basis functions | 46 KB |
| `normdata_eval.npz` | `NormData.scale_forward`/`scale_backward`; all Evaluator statistics plus BIC; `iter_batch_combinations`; `HBR.extract_and_reshape` | 82 KB |
| `blr.npz` | BLR inputs; `post`, `loglik`, `penalized_loglik`, `dloglik`, `ys_s2`, `forward`, `backward`, `elemwise_logp` at fixed hyperparameters; fitted hyperparameters and negative log-likelihood; predictions of the saved models; `sample_batch_effects` and `sample_covariates` with a fixed global seed | 82 KB |
| `blr_plain/`, `blr_warp/`, `blr_hetero/`, `blr_hetero_be/` | Saved BLR models (l-bfgs-b; no warp, SinhArcsinh warp, noise that changes with age, noise that changes with age and with the batch effects: `fixed_effect_var` and `fixed_effect_var_slope`) | 12 KB each |
| `hbr.npz` | HBR inputs and predictions (`compute_zscores`, `compute_centiles`, `compute_logp`, `compute_yhat`) of the saved models | 57 KB |
| `hbr_Normal/`, `hbr_SHASHb/`, `hbr_beta/`, `hbr_ZINB/` | Saved HBR models (pymc sampler, 2 chains, 20 tuning and 20 draws, fixed `random_seed`) | 23-45 KB each |
| `multi.npz` | Inputs with three response variables (`y0`, `y1`, `y2`, with clearly different means and scales); per response variable: fitted BLR negative log-likelihood and the predictions of both saved models. Keys end in `__<response variable>`, so the tests match by name, not position | 57 KB |
| `blr_multi/`, `hbr_multi/` | Saved plain BLR and Normal HBR models with three response variables (HBR sampler settings as above) | 25 KB, 117 KB |
| `predict.npz` | `NormativeModel.predict` as one call (with `evaluate_model` and `saveresults` on, plots off) of `blr_plain/` and `hbr_Normal/`: Z, centiles, `baseline_logp`, logp, Yhat, Evaluator statistics and their names, and the names of the result files | 18 KB |
| `provenance.json` | Commit, package versions, platform, seed and thread variables of the last run | 1 KB |

All stored inputs come from `np.random.default_rng` with seeds derived from
one master seed. The shared code is in `test/test_golden/_common.py`.

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
| Commit | `e932e3cdff4f08cfa74115a6924b372627c79b0f` (branch `gdevenyi/perf-benchmarks`, no changes under `pcntoolkit/`). The files of the earlier commit `fdbdbbb` were kept: the regeneration only added keys to `blr.npz` and added files; every older value is bit-identical |
| PCNtoolkit | 1.4.0 |
| Python | 3.13.12 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| PyMC / PyTensor / ArviZ / xarray | 6.3.2 / 3.3.2 / 1.3.0 / 2026.7.0 |
| Platform | Linux x86_64 (glibc 2.44) |
| Thread variables | none set |
| Total size | about 1.1 MB (was 0.5 MB before the multi-chunk, multi-response, `hetero_be` and `predict` additions) |
