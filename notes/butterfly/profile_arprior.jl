# Split the wall time of a nested sampling run on the arprior learned_hard_decay problem into
# likelihood time and sampler time. Usage: julia --project=<env> profile_arprior.jl P [seed]
using ARPrior, NestedSamplers, StableRNGs, Printf, LinearAlgebra
using AbstractMCMC: sample
include(joinpath(ENV["ARPRIOR"], "scripts", "feature_learning.jl"))
using .FeatureLearning
BLAS.set_num_threads(1)
const s = LEARN
P = parse(Int, ARGS[1]); seed = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
x = simulate_resonance()
stats = regression_stats(lag_design(x, P; start=s.start)...)
names, prior = models(P)[:learned_hard_decay]
all_names = (:logσ², :logλ, names...)
transform(u) = [FeatureLearning.box(u[i], getfield(s, all_names[i])) for i in eachindex(u)]
const TL = Ref(0.0); const NC = Ref(0)
function loglike(ψ)
    t = time_ns()
    v = log_evidence(EvidenceTerms(prior(NamedTuple{all_names}(Tuple(ψ))), stats), exp(ψ[1]), exp(ψ[2]))
    TL[] += (time_ns() - t) / 1e9; NC[] += 1
    v
end
model = NestedModel(loglike, transform)
spl() = Nested(5, 400; bounds=Bounds.MultiEllipsoid, enlarge=1.25, proposal=Proposals.Rejection(maxiter=10^7))
# warm up (compilation)
sample(StableRNG(99), model, spl(); dlogz=50.0, chain_type=Array, progress=false)
TL[] = 0; NC[] = 0
GC.gc()
stats_ = @timed sample(StableRNG(seed), model, spl(); dlogz=s.dlogz, chain_type=Array, progress=false)
(_, st) = stats_.value
@printf("P=%d seed=%d: wall %.2f s, likelihood %.2f s (%.0f%%) for %d calls (%.3f ms/call), sampler %.2f s (%.1f µs per iteration, %d iterations, %.1f calls/iteration), allocated %.2f GB, gc %.2f s\n",
        P, seed, stats_.time, TL[], 100TL[] / stats_.time, NC[], 1e3TL[] / NC[], stats_.time - TL[],
        1e6 * (stats_.time - TL[]) / st.it, st.it, NC[] / st.it, stats_.bytes / 1e9, stats_.gctime)
