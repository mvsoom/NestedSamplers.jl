# Diagnostic harness: instrumented NestedSamplers runs with insertion-index (rank) test,
# analytic-Z test problems, and a threaded driver over (problem, seed) jobs.
# Works with both the 0.9.0 baseline and the fork (uses only generic state fields).
using NestedSamplers, StableRNGs, Statistics, LinearAlgebra, Distributions, Random, Printf
using LogExpFunctions: logsumexp, logaddexp
import AbstractMCMC
using AbstractMCMC: step, samples, save!!, bundle_samples
LinearAlgebra.BLAS.set_num_threads(1)

"""Run one nested sampling run step by step, recording insertion indices of new points."""
function run_one(model, sampler; seed, dlogz=0.5, maxcall=10^8, record=true)
    rng = StableRNG(seed)
    t0 = time()
    smp, state = step(rng, model, sampler)
    sm = samples(smp, model, sampler)
    sm = save!!(sm, smp, 1, model, sampler)
    N = sampler.nactive
    ranks = Int[]
    calls = Int[]
    i = 1
    while !NestedSamplers.nested_isdone(rng, model, sampler, sm, state, i; progress=false, dlogz=dlogz, maxcall=maxcall)
        if record
            prev = copy(state.logl)
            idx = argmin(prev)
            nc0 = state.ncall
        end
        smp, state = step(rng, model, sampler, state)
        i += 1
        sm = save!!(sm, smp, i, model, sampler)
        if record
            lnew = state.logl[idx]
            push!(ranks, count(j -> j != idx && prev[j] < lnew, eachindex(prev)))
            push!(calls, state.ncall - nc0)
        end
    end
    ncall = state.ncall
    it = state.it
    vals, fstate = bundle_samples(sm, model, sampler, state, Array; check_wsum=false)
    (logz=fstate.logz, logzerr=fstate.logzerr, ncall=ncall, it=it, ranks=ranks, calls=calls,
     N=N, seconds=time() - t0)
end

"""Insertion-index statistics: z-score of the mean rank and KS distance / p-value against
the discrete uniform on 0:N-1 (Fowlie, Handley & Su 2020)."""
function rank_stats(ranks, N)
    n = length(ranks)
    n == 0 && return (z=NaN, D=NaN, p=NaN)
    r = (ranks .+ 0.5) ./ N            # in (0,1), uniform if correct
    z = (mean(r) - 0.5) / sqrt((1 - 1 / N^2) / 12 / n)
    counts = zeros(Int, N)
    for k in ranks; counts[k+1] += 1; end
    D = maximum(abs.(cumsum(counts) ./ n .- (1:N) ./ N))
    λ = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * D
    p = clamp(2 * sum((-1)^(k - 1) * exp(-2 * k^2 * λ^2) for k in 1:100), 0, 1)
    (z=z, D=D, p=p)
end

# ------------------------------------------------------------------ problems
# All on the unit cube (prior uniform), so Z = ∫ L(u) du over [0,1]^d.

"Log of the box mass of a diagonal Gaussian N(μ, diag(σ²)) over [0,1]^d."
logboxmass(μ, σ) = sum(log(cdf(Normal(m, s), 1) - cdf(Normal(m, s), 0)) for (m, s) in zip(μ, σ))

"Mixture of diagonal Gaussians: L(u) = Σ w_k N(u; μ_k, σ_k²). Z = Σ w_k boxmass_k."
function gaussmix(ws, μs, σs)
    lw = log.(ws)
    comps = [(lw[k], μs[k], σs[k], -sum(log.(σs[k])) - length(σs[k]) / 2 * log(2π)) for k in eachindex(ws)]
    function logl(u)
        acc = -Inf
        for (l, μ, σ, c) in comps
            s = 0.0
            @inbounds for i in eachindex(u)
                s += ((u[i] - μ[i]) / σ[i])^2
            end
            acc = logaddexp(acc, l + c - s / 2)
        end
        acc
    end
    lnZ = logsumexp([lw[k] + logboxmass(μs[k], σs[k]) for k in eachindex(ws)])
    NestedModel(logl, identity), lnZ
end

"Banana needle: unit-Jacobian shear y_n = u_n - a (u_1 - c)^2 of a diagonal Gaussian."
function banana(d; a=2.0, σ=fill(0.05, d), σn=0.004)
    σ = copy(σ); σ[end] = σn
    μ = fill(0.5, d); μ[end] = 0.4
    c = -sum(log.(σ)) - d / 2 * log(2π)
    function logl(u)
        s = 0.0
        @inbounds for i in 1:d-1
            s += ((u[i] - μ[i]) / σ[i])^2
        end
        y = u[d] - a * (u[1] - 0.5)^2
        s += ((y - μ[d]) / σ[d])^2
        c - s / 2
    end
    # analytic up to box truncation; checked by brute force below (tails are ≥ 8σ inside)
    NestedModel(logl, identity), 0.0
end

"Gaussian shell(s) of radius r and width w in [0,1]^d, centres cs. Z by 1-d radial quadrature."
function shells(d; r=0.2, w=0.005, cs=[fill(0.3, d), fill(0.7, d)])
    lnorm = -0.5 * log(2π * w^2)
    function logl(u)
        acc = -Inf
        for c in cs
            ρ = norm(u .- c)
            acc = logaddexp(acc, lnorm - (ρ - r)^2 / (2w^2))
        end
        acc
    end
    # ∫ N(ρ; r, w) S_d ρ^(d-1) dρ, S_d = surface of unit sphere; shells fully inside the cube
    Sd = 2π^(d / 2) / gamma(d / 2)
    ρs = range(r - 12w, r + 12w; length=20001); h = Base.step(ρs)
    I1 = sum(exp(lnorm - (ρ - r)^2 / (2w^2)) * Sd * ρ^(d - 1) for ρ in ρs) * h
    NestedModel(logl, identity), log(length(cs) * I1)
end
using SpecialFunctions: gamma

function problems()
    P = Dict{Symbol,Any}()
    # needle: two narrow coordinates (like logσ² and f), others broad
    P[:needle5] = gaussmix([1.0], [[0.5, 0.45, 0.55, 0.25, 0.5]], [[0.007, 0.04, 0.1, 0.007, 0.15]])
    # multimodal needle: three narrow peaks in coordinate 4 (like f), different weights/widths
    μb = [0.5, 0.45, 0.55, 0.0, 0.5]; σb = [0.007, 0.04, 0.1, 0.0, 0.12]
    mk(f, sf) = (m = copy(μb); m[4] = f; s = copy(σb); s[4] = sf; (m, s))
    comps = [mk(0.22, 0.007), mk(0.5, 0.02), mk(0.8, 0.004)]
    P[:mixneedle5] = gaussmix([0.6, 0.3, 0.1], first.(comps), last.(comps))
    # needle on a broad plateau (phase transition) in 5-d
    P[:plateau5] = gaussmix([0.98, 0.02], [[0.5, 0.45, 0.55, 0.25, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]],
                            [[0.007, 0.04, 0.1, 0.007, 0.15], [0.2, 0.2, 0.2, 0.2, 0.2]])
    # 3-d mixture of three unequal Gaussians
    P[:mix3] = gaussmix([0.5, 0.3, 0.2], [[0.2, 0.2, 0.2], [0.7, 0.7, 0.3], [0.3, 0.8, 0.75]],
                        [[0.02, 0.02, 0.02], [0.05, 0.01, 0.03], [0.005, 0.005, 0.005]])
    P[:shell2] = shells(2)
    P[:shell5] = shells(5; r=0.15, w=0.004, cs=[fill(0.3, 5), fill(0.72, 5)])
    P[:banana5] = banana(5)
    for D in (2, 4, 6, 10)
        P[Symbol(:corr, D)] = Models.CorrelatedGaussian(D)
    end
    dims = Dict(:needle5 => 5, :mixneedle5 => 5, :plateau5 => 5, :mix3 => 3, :shell2 => 2, :shell5 => 5,
                :banana5 => 5, :corr2 => 2, :corr4 => 4, :corr6 => 6, :corr10 => 10)
    Dict(k => (v..., dims[k]) for (k, v) in P)
end

"Run jobs [(name, model, lnZ, sampler_factory, seed)] threaded; returns a vector of NamedTuples."
function run_jobs(jobs; dlogz=0.5, maxcall=10^8, verbose=true)
    out = Vector{Any}(undef, length(jobs))
    lk = ReentrantLock()
    Threads.@threads :dynamic for k in eachindex(jobs)
        name, model, lnZ, mksampler, seed = jobs[k]
        r = try
            run_one(model, mksampler(); seed, dlogz, maxcall)
        catch e
            (logz=NaN, logzerr=NaN, ncall=-1, it=-1, ranks=Int[], calls=Int[], N=0, seconds=NaN, err=sprint(showerror, e))
        end
        rs = rank_stats(r.ranks, r.N)
        out[k] = (name=name, seed=seed, lnZ=lnZ, dz=r.logz - lnZ, logzerr=r.logzerr, ncall=r.ncall,
                  it=r.it, rz=rs.z, ksD=rs.D, ksp=rs.p, seconds=r.seconds)
        verbose && lock(lk) do
            @printf("%-12s seed %2d  dlnZ %+7.3f ± %.3f  ncall %8d  it %6d  rank z %+6.2f  KS p %.3g  %.1fs\n",
                    name, seed, r.logz - lnZ, r.logzerr, r.ncall, r.it, rs.z, rs.p, r.seconds)
            flush(stdout)
        end
    end
    out
end

function summarize(res)
    names = unique(r.name for r in res)
    println(rpad("problem", 12), "  nseeds  mean dlnZ ± sd     mean logzerr  z(mean)  mean rank-z  min KS p  mean ncall")
    for n in names
        rs = [r for r in res if r.name == n && isfinite(r.dz)]
        isempty(rs) && (println(n, ": all failed"); continue)
        dz = [r.dz for r in rs]; m = mean(dz); s = length(dz) > 1 ? std(dz) : NaN
        e = mean(r.logzerr for r in rs)
        @printf("%-12s  %6d  %+7.3f ± %.3f     %.3f       %+6.2f   %+6.2f      %.2g    %.0f\n", n, length(rs), m, s, e,
                m / (e / sqrt(length(rs))), mean(r.rz for r in rs), minimum(r.ksp for r in rs), mean(r.ncall for r in rs))
    end
end
