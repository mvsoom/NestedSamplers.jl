# Profile the sampler itself on a cheap analytic likelihood (5-d needle, Rejection + MultiEllipsoid).
using NestedSamplers, StableRNGs, Printf, Profile, LinearAlgebra
using AbstractMCMC: sample
BLAS.set_num_threads(1)
μ = [0.5, 0.45, 0.55, 0.25, 0.5]; σ = [0.007, 0.04, 0.1, 0.007, 0.15]
logl(u) = -0.5 * sum(abs2, (u .- μ) ./ σ)
model = NestedModel(logl, identity)
spl() = Nested(5, 400; bounds=Bounds.MultiEllipsoid, enlarge=1.25, proposal=Proposals.Rejection(maxiter=10^7))
run(seed) = sample(StableRNG(seed), model, spl(); dlogz=0.5, chain_type=Array, progress=false)
run(1)
ts = [(@timed run(s)) for s in 1:5]
for (s, t) in enumerate(ts)
    st = t.value[2]
    @printf("seed %d: %.3f s, %d it, %d calls, %.1f µs/it, %.2f GB alloc, gc %.3f s\n", s, t.time, st.it, st.ncall, 1e6t.time / st.it, t.bytes / 1e9, t.gctime)
end
if get(ENV, "PROFILE", "1") == "1"
    Profile.clear()
    Profile.init(n=10^7, delay=0.0005)
    @profile for s in 1:5; run(s); end
    Profile.print(IOContext(stdout, :displaysize => (2000, 220)); format=:flat, sortedby=:count, mincount=40, noisefloor=1)
end
