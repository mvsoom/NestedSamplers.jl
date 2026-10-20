# Usage: julia --project=<env> -t 10 run_analytic.jl <label> [problems...]
# Rejection + MultiEllipsoid at the arprior settings over 8 seeds per problem.
include(joinpath(@__DIR__, "harness.jl"))
using Serialization
label = ARGS[1]
P = problems()
names = length(ARGS) > 1 ? Symbol.(ARGS[2:end]) :
        [:needle5, :mixneedle5, :plateau5, :mix3, :shell2, :shell5, :banana5, :corr2, :corr4, :corr6]
nlive = parse(Int, get(ENV, "NLIVE", "400"))
enlarge = parse(Float64, get(ENV, "ENLARGE", "1.25"))
seeds = 1:parse(Int, get(ENV, "NSEEDS", "8"))
extra = eval(Meta.parse(get(ENV, "EXTRA", "(;)")))      # extra Nested kwargs, e.g. "(bootstrap=5,)"
mk(d) = () -> Nested(d, nlive; bounds=Bounds.MultiEllipsoid, enlarge=enlarge,
                     proposal=Proposals.Rejection(maxiter=10^7), extra...)
jobs = [(n, P[n][1], P[n][2], mk(P[n][3]), s) for n in names for s in seeds]
println("label=$label nlive=$nlive enlarge=$enlarge extra=$extra threads=$(Threads.nthreads())")
res = run_jobs(jobs)
println(); summarize(res)
serialize(joinpath(@__DIR__, "results_$(label).jls"), res)
