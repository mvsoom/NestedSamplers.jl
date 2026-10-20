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
        u, = propose(rng, spl.proposal, spl, bound, us, us[:, 1], -Inf, ALWAYS, 0.0)
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
        u, = propose(rng, spl.proposal, spl, bound, us, us[:, 1], -Inf, ALWAYS, 0.0)
        u ∈ left && u ∈ right
    end
    @test nboth / n ≈ frac atol = 0.03
end
