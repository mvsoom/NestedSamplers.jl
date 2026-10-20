# A/B: RSlice on mix3 with the proposal ellipsoid chosen by volume (fork, dynesty #364) or as the
# ellipsoid containing the start point (old). Usage: OLDAXES=0|1 julia -t 8 rslice_ab.jl
include(joinpath(@__DIR__, "harness.jl"))
if get(ENV, "OLDAXES", "0") == "1"
    @eval NestedSamplers.Bounds function rand_live(rng::AbstractRNG, me::MultiEllipsoid, us)
        idx = rand(rng, Base.axes(us, 2)); u = us[:, idx]
        idxs = findall(ell -> u ∈ ell, me.ellipsoids)
        isempty(idxs) && return u, nothing
        return u, me.ellipsoids[rand(rng, idxs)]
    end
    println("old axes: ellipsoid containing the start point")
end
P = problems()
nl = parse(Int, get(ENV, "NLIVE", "200")); sl = parse(Int, get(ENV, "SLICES", "5"))
jobs = [("mix3_RSlice$(sl)_n$nl", P[:mix3][1], P[:mix3][2], () -> Nested(3, nl; proposal=Proposals.RSlice(slices=sl)), s) for s in 1:12]
res = run_jobs(jobs; verbose=true); summarize(res)
