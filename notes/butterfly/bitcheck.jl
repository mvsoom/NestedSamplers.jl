# Bit-identity check for speed changes: run fixed seeds and print ln Z, logzerr, it, ncall and a hash
# of the whole output matrix. Compare the output before and after a change with `diff`.
include(joinpath(@__DIR__, "harness.jl"))
using AbstractMCMC: sample
P = problems()
cases = [(:needle5, () -> Nested(5, 400; proposal=Proposals.Rejection(maxiter=10^7))),
         (:banana5, () -> Nested(5, 400; proposal=Proposals.Rejection(maxiter=10^7))),
         (:mix3, () -> Nested(3, 300; proposal=Proposals.Rejection(maxiter=10^7), bootstrap=5, enlarge=1.0)),
         (:corr4, () -> Nested(4, 200; proposal=Proposals.RWalk())),
         (:mix3, () -> Nested(3, 200; proposal=Proposals.RSlice())),
         (:corr2, () -> Nested(2, 200; bounds=Bounds.Ellipsoid, proposal=Proposals.Slice()))]
for (name, mk) in cases, seed in 1:2
    vals, st = sample(StableRNG(seed), P[name][1], mk(); dlogz=0.5, chain_type=Array, progress=false)
    @printf("%-8s %-26s seed %d: logz %.15g  logzerr %.15g  it %d  ncall %d  hash %x\n", name,
            string(nameof(typeof(mk().proposal))), seed, st.logz, st.logzerr, st.it, st.ncall, hash(vals))
end
