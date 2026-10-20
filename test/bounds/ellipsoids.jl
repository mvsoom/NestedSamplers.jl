using NestedSamplers.Bounds: Ellipsoid, MultiEllipsoid, fit, scale!, decompose, volume, volume_prefactor

const NMAX = 20

@testset "ndims=$N" for N in 1:NMAX  
    @testset "Spheres" begin 
        scale = 5
        center = 2scale .* ones(N)
        A = Matrix(I, N, N) ./ scale^2
        ell = Ellipsoid(center, A)
        @test volume(ell) ≈ volume_prefactor(N) * scale^N
        axs, axlens = decompose(ell)
        @test axlens ≈ fill(scale, N)
        @test axs ≈ Bounds.axes(ell)
        @test norm.(eachcol(axs)) == fill(scale, N)
    end

    @testset "Scaling" begin
        scale = 1.5
        center = zeros(N)
        A = diagm(0 => rand(rng, N))
        ell = Ellipsoid(center, A)

        ell2 = Ellipsoid(center, A ./ scale^2)

        scale!(ell, scale^N)

        @test volume(ell) ≈ volume(ell2)
        @test ell.A ≈ ell2.A
        @test all(decompose(ell) .≈ decompose(ell2))
    end

    @testset "Contains" begin
        E = 1e-7
        ell = Ellipsoid(N)

        # Point just outside unit n-Spheres
        pt = (1 / √N + E) .* ones(N)
        @test pt ∉ ell

        # point just inside
        pt = (1 / √N - E) .* ones(N)
        @test pt ∈ ell

        A = diagm(0 => rand(rng, N))
        ell = Ellipsoid(zeros(N), A)

        for i in 1:N
            axlen = 1 / sqrt(A[i, i])
            pt = zeros(N)
            pt[i] = axlen + E
            @test pt ∉ ell
            pt[i] = axlen - E
            @test pt ∈ ell
        end
    end
end # testset

@testset "fit to fewer points than dimensions" begin
    # Late in a run a (multi-)ellipsoid cluster can hold fewer live points than dimensions, spread over a tiny
    # region. The covariance is then (nearly) singular; fitting used to break its symmetry and take a pseudo-
    # inverse with negative eigenvalues, so that `decompose` failed with a DomainError.
    for n in (5, 10, 20), s in (1e-3, 1e-5), logpointvol in (-5, -40)
        x = 0.5 .+ s .* randn(rng, 30, n)
        ell = Bounds.fit(Bounds.Ellipsoid, x; pointvol=exp(logpointvol))
        @test all(isfinite, ell.axlens) && all(>(0), ell.axlens)
        @test isfinite(Bounds.volume(ell))
        @test all(c -> c ∈ ell, eachcol(x))
        @test issymmetric(ell.A)
    end
end

@testset "MultiEllipsoid covers a curved region" begin
    # Uniform points in a sheared 5-d ball (a banana). The earlier split rule cut these 400 points
    # into about 20 clusters whose ellipsoids covered 87% of the region (71% at worst), which made
    # rejection sampling overestimate ln Z; the dynesty rule uses a few and covers > 99%.
    d, a = 5, 2.0
    s = [0.05, 0.05, 0.05, 0.05, 0.004]
    banana(n) = reduce(hcat, begin
        z = randn(rng, d); z .*= 3 .* rand(rng)^(1 / d) ./ norm(z)
        u = 0.5 .+ s .* z; u[d] += a * (u[1] - 0.5)^2; u
    end for _ in 1:n)
    for _ in 1:5
        bound = scale!(fit(MultiEllipsoid, banana(400)), 1.25)
        y = banana(4000)
        @test count(j -> y[:, j] ∈ bound, axes(y, 2)) / 4000 > 0.98
        @test length(bound) ≤ 6
    end

    # well-separated clusters are still split
    blob(c, n) = reduce(hcat, c .+ 0.05 .* NestedSamplers.Bounds.randball(rng, Float64, d) for _ in 1:n)
    @test length(fit(MultiEllipsoid, hcat(blob(0.3, 200), blob(0.7, 200)))) == 2
end

@testset "bootstrap expansion" begin
    # 30 uniform points in a 3-d ball: the fitted ellipsoid misses part of the ball; expanded by
    # the bootstrap factor it covers nearly all of it.
    ball(n) = reduce(hcat, NestedSamplers.Bounds.randball(rng, Float64, 3) for _ in 1:n)
    @test Bounds.bootstrap_expand(rng, Ellipsoid, ball(30), 0) == 1
    covered, covered_boot = Float64[], Float64[]
    for _ in 1:20
        x = ball(30)
        ell = fit(Ellipsoid, x)
        y = ball(2000)
        push!(covered, count(j -> y[:, j] ∈ ell, axes(y, 2)) / 2000)
        expand = Bounds.bootstrap_expand(rng, Ellipsoid, x, 5)
        @test expand ≥ 1
        scale!(ell, expand^3)
        push!(covered_boot, count(j -> y[:, j] ∈ ell, axes(y, 2)) / 2000)
    end
    @test mean(covered) < 0.95
    @test mean(covered_boot) > 0.99
    # distances: 1 on the surface of an ellipsoid, the nearest ellipsoid for a MultiEllipsoid
    ell = Ellipsoid([0.0, 0.0], diagm([1 / 4, 1.0]))
    @test Bounds.distance(ell, [2.0, 0.0]) ≈ 1
    @test Bounds.distance(MultiEllipsoid([ell, Ellipsoid([5.0, 0.0], diagm([1.0, 1.0]))]), [5.5, 0.0]) ≈ 0.5
end
