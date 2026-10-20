# How much does an ellipsoid fitted (Bounds.fit) to n uniform points in a d-ball under-cover the
# ball? Report mean coverage and the linear expansion needed for 99% / 99.9% coverage.
using NestedSamplers, StableRNGs, LinearAlgebra, Statistics, Printf
const B = NestedSamplers.Bounds
function ball(rng, d)
    z = randn(rng, d); z .* (rand(rng)^(1 / d) / norm(z))
end
rng = StableRNG(7)
for d in (2, 3, 5, 6, 8, 10), n in (2d, 4d, 10d, 40d, 400)
    n < 2d && continue
    covs = Float64[]; e99 = Float64[]; e999 = Float64[]
    for rep in 1:200
        x = reduce(hcat, ball(rng, d) for _ in 1:n)
        ell = B.fit(B.Ellipsoid, x)
        ds = sort([B.distance(ell, ball(rng, d)) for _ in 1:4000])
        push!(covs, mean(ds .<= 1)); push!(e99, ds[3960]); push!(e999, ds[3996])
    end
    @printf("d=%2d n=%4d  mean coverage %.4f  min %.3f  | expand for 99%%: median %.3f, 90%% %.3f (vol x%.2f) | 99.9%%: median %.3f\n",
            d, n, mean(covs), minimum(covs), median(e99), quantile(e99, 0.9), quantile(e99, 0.9)^d, median(e999))
end
