# Coverage of the true likelihood contour {L > L*} by the active bound, for the needle and
# banana problems, whose contours can be sampled exactly (a ball in whitened / sheared coords).
# Usage: ENLARGE=1.25 julia --project=<env> -t 8 coverage.jl [problem] [seeds]
include(joinpath(@__DIR__, "harness.jl"))
prob = length(ARGS) ≥ 1 ? Symbol(ARGS[1]) : :banana5
nseeds = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 4
enlarge = parse(Float64, get(ENV, "ENLARGE", "1.25"))
extra = eval(Meta.parse(get(ENV, "EXTRA", "(;)")))
d = 5
if prob == :needle5
    μ = [0.5, 0.45, 0.55, 0.25, 0.5]; σ = [0.007, 0.04, 0.1, 0.007, 0.15]; a = 0.0
else  # banana5 as in harness.jl
    μ = fill(0.5, d); μ[end] = 0.4; σ = fill(0.05, d); σ[end] = 0.004; a = 2.0
end
c = -sum(log.(σ)) - d / 2 * log(2π)
model = problems()[prob][1]

"Fraction of M uniform draws from {L > lstar} ∩ cube that lie in `bound`."
function coverage(bound, lstar, rng; M=4000)
    R = sqrt(2 * (c - lstar))
    inside = 0; total = 0
    while total < M
        z = randn(rng, d); z .*= R * rand(rng)^(1 / d) / norm(z)
        u = μ .+ σ .* z
        u[d] += a * (u[1] - 0.5)^2         # undo the shear (unit Jacobian)
        all(x -> 0 < x < 1, u) || continue
        total += 1
        inside += u ∈ bound
    end
    inside / total
end

res = Vector{Any}(undef, nseeds)
Threads.@threads for seed in 1:nseeds
    rng = StableRNG(seed); crng = StableRNG(1000 + seed)
    spl = Nested(d, 400; bounds=Bounds.MultiEllipsoid, enlarge=enlarge, proposal=Proposals.Rejection(maxiter=10^7), extra...)
    smp, state = step(rng, model, spl)
    covs = Float64[]; nells = Int[]
    while !NestedSamplers.nested_isdone(rng, model, spl, nothing, state, state.it; progress=false, dlogz=0.5)
        smp, state = step(rng, model, spl, state)
        if state.has_bounds && state.it % 25 == 0
            b = state.active_bound
            push!(covs, coverage(b, minimum(state.logl), crng))
            push!(nells, b isa Bounds.MultiEllipsoid ? length(b) : 1)
        end
    end
    res[seed] = (covs=covs, nells=nells, ncall=state.ncall, it=state.it)
end
println("problem=$prob enlarge=$enlarge extra=$extra")
for (s, r) in enumerate(res)
    q = sort(r.covs)
    @printf("seed %d: checks %d  mean coverage %.4f  min %.3f  5%% quantile %.3f  frac<0.99 %.2f  frac<0.9 %.2f  mean #ells %.1f  ncall %d\n",
            s, length(q), mean(q), q[1], q[max(1, round(Int, 0.05length(q)))], mean(q .< 0.99), mean(q .< 0.9), mean(r.nells), r.ncall)
end
