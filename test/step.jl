using NestedSamplers: propose
import Random

# disk of radius r around c, as a 2-d ellipsoid
disk(c, r) = Bounds.Ellipsoid(c, diagm(fill(1 / r^2, 2)))
const ALWAYS = NestedModel(x -> 0.0, identity)  # flat likelihood: every draw is accepted

@testset "Rejection draws uniformly from the union of ellipsoids" begin
    # Two disjoint disks with areas 1:4 holding equal numbers of live points. Drawing from the
    # ellipsoid around a random live point (the old behaviour) splits the draws 50:50 and biased
    # ln Z upward; uniform draws from the union split them 20:80.
    small, big = disk([0.25, 0.25], 0.1), disk([0.7, 0.7], 0.2)
    bound = Bounds.MultiEllipsoid([small, big])
    us = hcat(reduce(hcat, rand(rng, small) for _ in 1:50), reduce(hcat, rand(rng, big) for _ in 1:50))
    spl = Nested(2, 100; proposal=Proposals.Rejection())
    n = 4000
    nsmall = count(1:n) do _
        u, = propose(rng, spl.proposal, spl, bound, us, zeros(size(us, 2)), us[:, 1], -Inf, ALWAYS, 0.0)
        u ∈ small
    end
    @test nsmall / n ≈ 0.2 atol = 0.03

    # Overlapping disks: the overlap must not be counted twice (the 1/q acceptance).
    r, d = 0.2, 0.2
    left, right = disk([0.5 - d / 2, 0.5], r), disk([0.5 + d / 2, 0.5], r)
    bound = Bounds.MultiEllipsoid([left, right])
    lens = 2r^2 * acos(d / 2r) - d / 2 * sqrt(4r^2 - d^2)
    frac = lens / (2π * r^2 - lens)   # 0.243; without the correction 0.391
    nboth = count(1:n) do _
        u, = propose(rng, spl.proposal, spl, bound, us, zeros(size(us, 2)), us[:, 1], -Inf, ALWAYS, 0.0)
        u ∈ left && u ∈ right
    end
    @test nboth / n ≈ frac atol = 0.03
end

# records where an MCMC proposal starts and which bound it gets
struct StartRecorder <: Proposals.AbstractProposal end
(::StartRecorder)(rng, point, logl_star, bound, model) = (copy(point), bound, 0.0, 0)

@testset "walk and slice proposals: start point and axes" begin
    small, big = disk([0.25, 0.25], 0.1), disk([0.7, 0.7], 0.2)
    bound = Bounds.MultiEllipsoid([small, big])
    us = reduce(hcat, rand(rng, small) for _ in 1:20)   # all live points in the small disk
    logl = collect(1.0:20.0)                            # point 1 is the dead point
    spl = Nested(2, 20; proposal=Proposals.RWalk())
    n = 2000
    draws = [propose(rng, StartRecorder(), spl, bound, us, logl, us[:, 1], logl[1], ALWAYS, 0.0) for _ in 1:n]
    # never start from the dead point, which sits on the likelihood contour
    @test !any(d -> d[1] == us[:, 1], draws)
    # the ellipsoid that sets the proposal axes is chosen by volume (1:4), not by containing the
    # start point: axes that depend on the start point break detailed balance (dynesty #364)
    @test count(d -> d[2] === big, draws) / n ≈ 0.8 atol = 0.03
end

@testset "bounds are refitted every update_interval calls" begin
    # `since_update % update_interval == 0` rarely held once iterations cost more than one call,
    # so bounds went stale (one run: 406 143 calls between refits for an interval of 600) and
    # could be fitted before the min_ncall / min_eff criteria were met.
    model, _ = Models.CorrelatedGaussian(3)
    spl = Nested(3, 200; proposal=Proposals.Rejection())
    srng = StableRNG(2)
    _, state = AbstractMCMC.step(srng, model, spl)
    first_ncall, last, gaps, prev = 0, 0, Int[], nothing
    for _ in 1:3000
        _, state = AbstractMCMC.step(srng, model, spl, state)
        if state.has_bounds && state.active_bound !== prev
            first_ncall == 0 ? (first_ncall = state.ncall) : push!(gaps, state.ncall - last)
            last, prev = state.ncall, state.active_bound
        end
    end
    @test first_ncall > spl.min_ncall
    @test length(gaps) ≥ 10
    @test maximum(gaps) < spl.update_interval + 100   # one iteration costs far fewer calls here
end

# unit-cube bound that records the `pointvol` it is fitted with
struct RecordingBound <: Bounds.AbstractBoundingSpace{Float64}
    ndims::Int
end
const RECORDED_POINTVOLS = Float64[]
function NestedSamplers.Bounds.fit(::Type{RecordingBound}, x::AbstractMatrix; pointvol = 0)
    push!(RECORDED_POINTVOLS, pointvol)
    return RecordingBound(size(x, 1))
end
NestedSamplers.Bounds.scale!(b::RecordingBound, factor) = b
Base.rand(rng::Random.AbstractRNG, b::RecordingBound) = rand(rng, b.ndims)
Base.in(x::AbstractVector, ::RecordingBound) = true
Base.ndims(b::RecordingBound) = b.ndims

@testset "bounds are fitted with the prior volume per live point" begin
    # `pointvol` sets the minimum volume of a fitted ellipsoid (npoints * pointvol) and the split
    # test of MultiEllipsoid; it was X / nactive^2 instead of X / nactive.
    model = NestedModel(x -> -sum(abs2, x .- 0.5) / 0.02, identity)
    nlive = 100
    spl = Nested(2, nlive; bounds=RecordingBound, proposal=Proposals.Rejection(), min_eff=1.0)
    srng = StableRNG(3)
    empty!(RECORDED_POINTVOLS)
    _, state = AbstractMCMC.step(srng, model, spl)
    checked = 0
    for _ in 1:500   # the bound stays the unit cube, so keep X = exp(-it / nlive) moderate
        it, logvol, nrec = state.it, state.logvol, length(RECORDED_POINTVOLS)
        _, state = AbstractMCMC.step(srng, model, spl, state)
        @test logvol ≈ -it * log((nlive + 1) / nlive)   # ln X, shrinking by N / (N + 1) per iteration
        if length(RECORDED_POINTVOLS) > nrec
            @test last(RECORDED_POINTVOLS) ≈ exp(logvol) / nlive
            checked += 1
        end
    end
    @test checked ≥ 3
end

@testset "bootstrap-expanded bounds" begin
    model, lnZ = Models.CorrelatedGaussian(3)
    spl = Nested(3, 300; bootstrap=5, enlarge=1.0)
    @test spl.bootstrap == 5
    _, state = sample(StableRNG(5), model, spl; dlogz=0.1, progress=false, chain_type=Array)
    @test state.logz ≈ lnZ atol = 4state.logzerr
end

@testset "evidence integration" begin
    # one trapezoid step: shell between X = 1 and X = N / (N + 1) with L = 1 and L = 3
    N = 10
    dlv = log((N + 1) / N)
    logwt, logz, h = NestedSamplers.integrate(0.0, log(3), -dlv, dlv, -Inf, 0.0)
    @test exp(logwt) ≈ (1 + 3) / 2 * (1 - N / (N + 1))
    @test logz ≈ logwt
    # H = ∫ L log L dX / Z - ln Z for the trapezoid shell
    @test h ≈ (log(3) * 3 / 4) - logz
    # zero likelihood at both edges: weight 0 and finite information
    logwt, logz, h = NestedSamplers.integrate(-1e300, -1e300, -dlv, dlv, -Inf, 0.0)
    @test h == 0

    # the remaining live points are added in order of increasing likelihood
    model, lnZ = Models.CorrelatedGaussian(2)
    spl = Nested(2, 50)
    srng = StableRNG(9)
    smp, state = AbstractMCMC.step(srng, model, spl)
    samples = AbstractMCMC.samples(smp, model, spl)
    samples = AbstractMCMC.save!!(samples, smp, 1, model, spl)
    for i in 2:200
        smp, state = AbstractMCMC.step(srng, model, spl, state)
        samples = AbstractMCMC.save!!(samples, smp, i, model, spl)
    end
    samples, final = NestedSamplers.add_live_points(samples, model, spl, state)
    added = [s.logl for s in samples[201:end]]
    @test length(added) == 50 && issorted(added)
    @test final.it == 250
    @test sum(s -> exp(s.logwt - final.logz), samples) ≈ 1 atol = 1e-8
end
