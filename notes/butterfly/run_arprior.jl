# Usage: MODELS=learned_hard_decay ORDERS=12,48 NSEEDS=4 julia --project=<env> -t 8 run_arprior.jl <label>
# Instrumented nested sampling on arprior's feature-learning models (read-only use of arprior code).
using ARPrior, Serialization
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(ENV["ARPRIOR"], "scripts", "feature_learning.jl"))
using .FeatureLearning
const s = LEARN
const QUAD = Dict((12, :base_decay) => -689.587, (48, :base_decay) => -689.934,
                  (12, :fixed_hard_decay) => -683.472, (48, :fixed_hard_decay) => -683.239,
                  (12, :learned_hard_decay) => -686.8273, (48, :learned_hard_decay) => -687.2110,
                  (12, :learned_moment_decay) => -687.407, (48, :learned_moment_decay) => -687.804,
                  (12, :learned_hard_iso) => -691.414, (48, :learned_hard_iso) => -725.375)

function arprior_model(P, name; stats)
    names, prior = models(P)[name]
    all_names = (:logσ², :logλ, names...)
    transform(u) = [FeatureLearning.box(u[i], getfield(s, all_names[i])) for i in eachindex(u)]
    loglike(ψ) = log_evidence(EvidenceTerms(prior(NamedTuple{all_names}(Tuple(ψ))), stats), exp(ψ[1]), exp(ψ[2]))
    NestedModel(loglike, transform), length(all_names)
end

label = ARGS[1]
mnames = Symbol.(split(get(ENV, "MODELS", "learned_hard_decay"), ','))
orders = parse.(Int, split(get(ENV, "ORDERS", "12,48"), ','))
seeds = parse(Int, get(ENV, "SEED0", "1")) .+ (0:parse(Int, get(ENV, "NSEEDS", "4"))-1)
nlive = parse(Int, get(ENV, "NLIVE", "400"))
enlarge = parse(Float64, get(ENV, "ENLARGE", "1.25"))
extra = eval(Meta.parse(get(ENV, "EXTRA", "(;)")))
x = simulate_resonance()
stats = Dict(P => regression_stats(lag_design(x, P; start=s.start)...) for P in orders)
jobs = []
for P in orders, m in mnames
    model, d = arprior_model(P, m; stats=stats[P])
    for seed in seeds
        push!(jobs, ("P$(P)_$(m)", model, QUAD[(P, m)],
                     () -> Nested(d, nlive; bounds=Bounds.MultiEllipsoid, enlarge=enlarge,
                                  proposal=Proposals.Rejection(maxiter=10^7), extra...), seed))
    end
end
println("label=$label nlive=$nlive enlarge=$enlarge extra=$extra threads=$(Threads.nthreads())")
res = run_jobs(jobs; dlogz=s.dlogz)
println(); summarize(res)
serialize(joinpath(@__DIR__, "arprior_$(label).jls"), res)
