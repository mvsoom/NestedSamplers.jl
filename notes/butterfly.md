# Branch `butterfly`: correctness and speed of NestedSamplers.jl (October 2026)

Local work on top of `fix-ellipsoid-eigenvalues` (upstream PRs #124, #125). Not pushed.
Started from an overestimate of ln Z by 0.4 to 0.8 nats in the arprior feature-learning
models (Rejection + MultiEllipsoid, nlive 400, enlarge 1.25, dlogz 0.5).

## Commits (oldest first)

| commit | change |
|---|---|
| 64feb61 | Rejection draws from the whole bound (union of ellipsoids, 1/q overlap correction), not from the one ellipsoid around a random live point |
| b834221 | Walk/slice proposals: proposal ellipsoid chosen by volume (dynesty #364); start from a live point above the threshold |
| f9feb13 | Refit bounds every `update_interval` calls (`≥`, not `%`); no bound before the min_ncall/min_eff criteria |
| 4b9bd67 | `pointvol` = X / nactive (was X / nactive²) |
| 21d3655 | MultiEllipsoid splitting rule of dynesty 2.1.5 (BIC-like volume penalty, k-means in scaled coordinates) |
| fa67e73 | Optional `bootstrap` expansion (dynesty), default off |
| b6fb996 | Trapezoid integrator of PR #80 (Miles Lucas) with sorted final live points, stable H (Jose Vines, normalized), Skilling logzerr |
| ccb1bfb | Zero-likelihood plateaus: dynesty's plateau volumes; strict `>` acceptance |
| 8526add | Output matrix built in place (was quadratic `mapreduce(vcat)`) |
| 53f6576 | Ellipsoid fit reuses the eigendecomposition |
| c5b33da | MultiEllipsoid caches its volume weights |
| 273e513 | `ncall` counts the initial live points and only real evaluations |

Every commit message has its evidence. Speed commits are bit-identical (`bitcheck.jl`).

## Root causes of the arprior overestimate

1. **Rejection sampled one ellipsoid, chosen through a random live point** (64feb61). Ellipsoids
   were weighted by their share of live points, not their volume, and overlaps counted twice.
   Unit test: two disks with areas 1:4 and 50 live points each get 50:50 of the draws (should be
   20:80). This was most of the bias: arprior learned_hard_decay +0.71 / +0.80 → +0.23 / +0.11.
2. **Bounds cut off part of the likelihood contour** (21d3655, with 4b9bd67). An ellipsoid fitted
   to n uniform points of an ellipsoidal region covers on average (`ellcover.jl`, d = 5):
   34% for n = 10, 71% for n = 20, 93% for n = 50, 99% for n = 400. The old splitting rule cut
   curved contours into ~20 clusters of ~20 points; on a 5-d banana the bound covered 84 to 93%
   of the true contour (`coverage.jl`). With dynesty's rule: 2 to 6 ellipsoids, 99.6 to 99.7%.
   Remaining arprior bias after (1) and the schedule fix: +0.20 / +0.31; after (2): ≈ 0.
3. The update schedule (f9feb13) did not bias ln Z but made calls erratic (needle: 18k to 441k
   calls per run, one gap of 406k calls between refits). Stale bounds partly hid (2).

Ruled out as causes of the arprior bias: the integrator and final live-point contribution
(the trapezoid port changed ln Z by < 0.02 on every test problem), the logzerr estimate (seed
scatter matches sqrt(H/N)), the first update.

Also found and fixed, not relevant to arprior: zero-likelihood regions biased ln Z by about +0.19
(package's own "Zero likelihood" test model, 5 to 7 sigma) through the tied -1e300 live points
(ccb1bfb); `mapreduce(vcat)` in `bundle_samples` was 87% of the sampler's time on cheap problems.

## Validation tables (ln Z minus truth, mean ± sd over seeds)

Analytic problems on the unit cube (`harness.jl`, `run_analytic.jl`), Rejection + MultiEllipsoid,
nlive 400, enlarge 1.25, dlogz 0.5, 8 seeds (StableRNG 1 to 8). "z" is mean / (logzerr / √8);
"rank z" the mean z-score of insertion indices (Fowlie, Handley & Su 2020).

| problem | truth | 0.9.0 | z | rank z | butterfly | z | rank z | calls 0.9.0 → butterfly |
|---|---|---|---|---|---|---|---|---|
| 5-d needle (σ 0.007 in two coordinates) | -0.0009 | +0.088 ± 0.211 | +1.6 | +0.4 | +0.034 ± 0.205 | +0.6 | +0.7 | 185k → 24k |
| 5-d three-mode needle | 0.0 | +0.187 ± 0.206 | +3.5 | +0.7 | +0.024 ± 0.072 | +0.4 | -0.3 | 200k → 57k |
| 5-d needle on a plateau | -0.0021 | +0.054 ± 0.293 | +1.0 | 0.0 | -0.011 ± 0.140 | -0.2 | -0.2 | 141k → 44k |
| 3-d mixture of unequal Gaussians | 0.0 | -0.027 ± 0.165 | -0.6 | +0.2 | -0.012 ± 0.192 | -0.3 | -0.3 | 159k → 28k |
| 2-d Gaussian shells | 0.9216 | -0.016 ± 0.083 | -0.5 | +0.4 | -0.019 ± 0.087 | -0.6 | +0.3 | 11k → 15k |
| 5-d Gaussian shells | -3.6208 | -0.026 ± 0.179 | -0.5 | +0.7 | +0.058 ± 0.076 | +1.2 | +0.7 | 53k → 25k |
| 5-d banana needle | 0.0 | **+0.502 ± 0.400** | **+9.0** | **+3.7** | -0.024 ± 0.183 | -0.4 | +0.2 | 83k → 89k |
| CorrelatedGaussian(2) | -3.7591 | -0.008 ± 0.071 | -0.4 | -0.2 | +0.012 ± 0.065 | +0.6 | -0.1 | 4.8k → 14.6k |
| CorrelatedGaussian(4) | -6.1879 | +0.010 ± 0.110 | +0.3 | +0.1 | +0.043 ± 0.068 | +1.3 | -0.1 | 30k → 17k |
| CorrelatedGaussian(6) | -8.3682 | +0.051 ± 0.130 | +1.2 | 0.0 | +0.045 ± 0.080 | +1.1 | +0.1 | 49k → 21k |

arprior feature-learning models (`run_arprior.jl`, arprior at 64e282a, data `simulate_resonance()`),
same sampler settings; truth = converged tensor quadrature from arprior's `quadrature.tsv`.
0.9.0 values from arprior's `results/runs/feature_learning/runs.tsv` (seeds 1 to 4).

| P | model | quadrature | 0.9.0 (4 seeds) | butterfly | seeds | z | calls 0.9.0 → butterfly (mean) |
|---|---|---|---|---|---|---|---|
| 12 | base_decay | -689.587 | +0.046 ± 0.117 | -0.007 ± 0.167 | 4 | -0.1 | 349k → 20k |
| 12 | fixed_hard_decay | -683.472 | +0.083 ± 0.105 | -0.092 ± 0.234 | 4 | -1.5 | 36k → 20k |
| 12 | learned_hard_decay | -686.827 | **+0.705 ± 0.271** | +0.017 ± 0.128 | 8 | +0.3 | 50k → 46k |
| 12 | learned_moment_decay | -687.407 | **+0.383 ± 0.211** | +0.086 ± 0.241 | 4 | +1.1 | 1.44M → 53k |
| 12 | learned_hard_iso | -691.414 | +0.150 ± 0.249 | -0.054 ± 0.208 | 4 | -0.7 | 560k → 36k |
| 48 | base_decay | -689.934 | +0.039 ± 0.048 | +0.002 ± 0.172 | 4 | 0.0 | 101k → 20k |
| 48 | fixed_hard_decay | -683.239 | -0.058 ± 0.112 | -0.005 ± 0.210 | 4 | -0.1 | 32k → 20k |
| 48 | learned_hard_decay | -687.211 | **+0.802 ± 0.249** | +0.039 ± 0.102 | 8 | +0.7 | 64k → 43k |
| 48 | learned_moment_decay | -687.804 (coarse) | **+0.470 ± 0.230** | -0.014 ± 0.208 | 4 | -0.2 | 81k → 55k |
| 48 | learned_hard_iso | -725.375 | +0.223 ± 0.245 | +0.101 ± 0.258 | 4 | +1.4 | 42k → 31k |

butterfly logzerr: 0.13 (3-d models) to 0.16 (5- and 6-d). One run, P = 48 learned_hard_decay seed 2,
fails the rank test (KS p = 0.0007, rank z -2.8), localized to iterations 1201 to 2000, just after
the first bound; six other seeds show no shift in that window (`rankwindow.jl`). Its ln Z is +0.14. Unexplained.

dynesty 2.1.5 on the same likelihood (`dyn_arprior.py`: independent numpy implementation, matches
Julia to 1e-11; `bound='multi', sample='unif'`, i.e. bootstrap 5; nlive 400, dlogz 0.5):

| P | quadrature | dynesty seeds 1, 2, 3 | mean − quadrature | butterfly (8 seeds) | calls dynesty |
|---|---|---|---|---|---|
| 12 | -686.827 | -686.762, -686.839, -687.050 (± 0.30 reported) | -0.06 ± 0.15 | +0.02 ± 0.13 | 148k, 89k, 76k |
| 48 | -687.211 | -687.246, -687.169, -687.326 (± 0.30 reported) | -0.04 ± 0.08 | +0.04 ± 0.10 | 110k, 88k, 106k |

dynesty and the fork agree within errors (difference +0.08 at both orders; dynesty sem about 0.09).
dynesty warns that the bootstrap enlargement is very large on this problem. Its reported logzerr
(0.30) is about twice sqrt(H/N): `compute_integrals` sums ΔH Δln X, which the final live points
inflate at dlogz 0.5.

## Speed

arprior learned_hard_decay, `profile_arprior.jl` (likelihood timed inside the call; "sampler" is
the rest, including the prior transform and GC triggered by the likelihood), one process at a
time, BLAS 1 thread, seeds 1 and 2, on a machine with load 15 to 25:

| P | version | wall (s) | likelihood share | ms per call | calls per iteration | sampler µs per iteration |
|---|---|---|---|---|---|---|
| 12 | 0.9.0 | 3.90, 2.09 | 68%, 49% | 0.042 | 11.2, 4.1 | 221, 186 |
| 12 | butterfly | 2.52, 2.21 | 76%, 77% | 0.043 | 7.4, 6.8 | 100, 87 |
| 48 | 0.9.0 | 21.6, 35.2 | 94%, 95% | 0.354 | 10.2, 16.6 | 227, 340 |
| 48 | butterfly | 14.9, 15.3 | 96%, 96% | 0.342 | 7.0, 7.2 | 99, 101 |

The likelihood is 0.04 ms (P = 12) and 0.35 ms (P = 48) per call here, not 0.3 to 1.5 ms. The
sampler itself (5-d needle with a microsecond likelihood, `profile_sampler.jl`) went from 150 to
190 µs per iteration to 21 to 31; 87% of the old time was the quadratic output assembly. What is
left is mostly bound refits (k-means, eigendecompositions; every 600 calls) and allocation in
`rand`/`in` for ellipsoids. Calls per iteration also became steady (0.9.0 runs of the arprior
models took up to 4.4M calls; now 20k to 55k).

## Things to know

- `bootstrap = 5` (dynesty's default for uniform sampling) is unstable on needle-shaped multimodal
  problems: one global factor, inflated by a small cluster (factor 19, volume x 2.4e6), stalled 5 of
  8 runs on the three-mode needle. dynesty itself warns about it on arprior. Default stays 0.
- MCMC proposals (RWalk, RSlice) on well-separated modes of very different widths have excess
  variance (3-d mixture, RSlice(5), nlive 200: sd 0.49 against logzerr 0.19 over 12 seeds), with
  either axis rule. Not a bias.
- Plateau runs: logzerr does not include the binomial error of the plateau fraction.

## How to rerun

```sh
# scratch environment (do not touch arprior's own Project/Manifest)
julia +1.12 -e 'import Pkg; Pkg.activate("ns-fork"); Pkg.develop([Pkg.PackageSpec(path="<this repo>"), Pkg.PackageSpec(path="<arprior checkout at 64e282a>")]); Pkg.add(["StableRNGs","Distributions","Statistics","SpecialFunctions","Serialization","LogExpFunctions","AbstractMCMC","DSP"])'
cd notes/butterfly
julia +1.12 --project=<env> -t 8 run_analytic.jl label                     # analytic table (8 seeds)
ARPRIOR=<arprior> MODELS=learned_hard_decay ORDERS=12,48 NSEEDS=8 julia +1.12 --project=<env> -t 8 run_arprior.jl label
ENLARGE=1.25 julia +1.12 --project=<env> -t 4 coverage.jl banana5 4      # contour coverage of the bounds
julia +1.12 --project=<env> ellcover.jl                                    # ellipsoid fit coverage vs n, d
julia +1.12 --project=<env> bitcheck.jl > out.txt                          # bit-identity check for speed work
ARPRIOR=<arprior> julia +1.12 --project=<env> profile_arprior.jl 12 1     # likelihood vs sampler time
OMP_NUM_THREADS=1 python -I dyn_arprior.py data.txt loglike_checks.tsv 48 unif 400 0.5 1 2
```
Environment variables of the drivers: NLIVE, ENLARGE, NSEEDS, SEED0, EXTRA (extra `Nested` keywords,
e.g. `EXTRA="(bootstrap=5,)"`). The full test suite: `julia +1.12 --project=. -e 'import Pkg; Pkg.test()'`
(about 8 minutes on a loaded machine; all 4536 tests pass at 273e513).

## Upstream survey (chalk-lab/NestedSamplers.jl, not moved; TuringLang URL redirects there)

- #80 integrator rewrite (Miles Lucas): ported, with a fix (live points sorted) and without
  dynesty's variance sum.
- #81 relative eigenvalue threshold (aplavin): superseded by #125 (condition number cap 1e12).
- jvines/NestedSamplers.jl `fix/high-dim-stability` (no PR): stable H recursion ported; split-depth
  cap, looser "stuck" thresholds and a batch-parallel sampler (approximate, O(K/N) bias) not ported.
- #103 (maintainer wanted), #33 performance, #70 parallel, #75, #105 Turing interface, #109 model:
  no correctness content. jvines reported "a bug with the multi ellipsoid bounds" in #103 without details.
