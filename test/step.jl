using NestedSamplers: propose

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
