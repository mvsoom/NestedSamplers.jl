# Insertion-rank z-score in windows of 400 iterations for one arprior run, plus bound size.
using ARPrior
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(ENV["ARPRIOR"], "scripts", "feature_learning.jl"))
using .FeatureLearning
const s = LEARN
P = parse(Int, ARGS[1]); seed = parse(Int, ARGS[2])
x = simulate_resonance()
stats = regression_stats(lag_design(x, P; start=s.start)...)
names, prior = models(P)[:learned_hard_decay]
all_names = (:logσ², :logλ, names...)
transform(u) = [FeatureLearning.box(u[i], getfield(s, all_names[i])) for i in eachindex(u)]
loglike(ψ) = log_evidence(EvidenceTerms(prior(NamedTuple{all_names}(Tuple(ψ))), stats), exp(ψ[1]), exp(ψ[2]))
model = NestedModel(loglike, transform)
spl = Nested(5, 400; bounds=Bounds.MultiEllipsoid, enlarge=1.25, proposal=Proposals.Rejection(maxiter=10^7))
function trace(model, spl, seed)
rng = StableRNG(seed)
smp, state = step(rng, model, spl)
ranks = Int[]; nells = Int[]; lstar = Float64[]
while !NestedSamplers.nested_isdone(rng, model, spl, nothing, state, state.it; progress=false, dlogz=0.5)
    prev = copy(state.logl); idx = argmin(prev)
    smp, state = step(rng, model, spl, state)
    push!(ranks, count(j -> j != idx && prev[j] < state.logl[idx], eachindex(prev)))
    push!(nells, state.has_bounds ? length(state.active_bound) : 0); push!(lstar, prev[idx])
end
return ranks, nells, lstar, state
end
ranks, nells, lstar, state = trace(model, spl, seed)
println("P=$P seed=$seed it=$(state.it) whole-run ", rank_stats(ranks, 400))
for w in 1:400:length(ranks)
    r = ranks[w:min(end, w + 399)]
    st = rank_stats(r, 400)
    @printf("iterations %5d-%5d  logL* %9.2f  #ells %4.1f  rank z %+5.2f  KS p %.3g\n", w, w + length(r) - 1, lstar[w],
            mean(nells[w:min(end, w + 399)]), st.z, st.p)
end
