using NestedSamplers, StableRNGs, LinearAlgebra, Statistics, Printf
const B = NestedSamplers.Bounds
# uniform points in a sheared ball (unit Jacobian): x_d += a (x_1 - 0.5)^2
function banana(rng, d, n; a=2.0, σ=0.05, σn=0.004)
    s = fill(σ, d); s[end] = σn
    reduce(hcat, begin
        z = randn(rng, d); z .*= rand(rng)^(1 / d) / norm(z)
        u = 0.5 .+ 3 .* s .* z; u[d] += a * (u[1] - 0.5)^2; u
    end for _ in 1:n)
end
rng = StableRNG(11)
for d in (2, 3, 5)
    cov = Float64[]; ne = Int[]
    for rep in 1:20
        x = banana(rng, d, 400)
        b = B.scale!(B.fit(B.MultiEllipsoid, x; pointvol = 0), 1.25)
        y = banana(rng, d, 4000)
        push!(cov, mean(y[:, j] ∈ b for j in axes(y, 2))); push!(ne, length(b))
    end
    @printf("d=%d: coverage mean %.4f min %.4f   #ellipsoids mean %.1f\n", d, mean(cov), minimum(cov), mean(ne))
end
