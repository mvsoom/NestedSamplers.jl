
@testset "Bundles" begin
    logl(x::AbstractVector) =  exp(-x[1]^2 / 2) / √(2π)
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    chains, _ = sample(rng, model, spl; dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr, _ = sample(rng, model, spl; dlogz=0.2, chain_type=Array)

    @test size(chains, 2) == size(val_arr, 2)

    # test with add_live = false
    chains2, _ = sample(rng, model, spl; add_live=false, dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr2, _ = sample(rng, model, spl; add_live=false, dlogz=0.2, chain_type=Array)
    
    @test size(chains2, 2) == size(val_arr2, 2)
    @test size(chains2, 1) < size(chains, 1) && size(val_arr2, 1) < size(val_arr, 1)

    # test check_wsum kwarg
    chains3, _ = sample(rng, model, spl; dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr3, _ = sample(rng, model, spl; dlogz=0.2, chain_type=Array)

    @test size(chains3, 2) == size(val_arr3, 2)
end

@testset "Zero likelihood" begin
    logl(x::AbstractVector) = x[1] > 0 ? exp(-x[1]^2 / 2) / √(2π) : -Inf
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    chains, state = sample(rng, model, spl; param_names=["x"])
    @test all(>(0), chains[:x][chains[:weights] .> 1e-10])
    # live points at the -1e300 stand-in for log(0) must not blow up the information
    @test 0 < state.logzerr < 0.1
    # The initial live points with zero likelihood form a plateau, which takes a fraction
    # n / (N + 1) of the prior volume. Removing them one by one with the usual geometric
    # shrinkage gave ln Z 0.17 to 0.22 too high here (5 to 7 sigma).
    lnZ = -0.35064   # log ∫_0^1 exp(φ(x)) dx / 2, by quadrature
    for seed in 1:3
        _, st = sample(StableRNG(seed), model, spl; progress=false, chain_type=Array)
        @test st.logz ≈ lnZ atol = 3st.logzerr
    end
end

@testset "Stopping criterion" begin
    logl(x::AbstractVector) =  exp(-x[1]^2 / 2) / √(2π)
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    
    chains, state = sample(rng, model, spl; add_live=false, dlogz=1.0)
    logz_remain = maximum(state.logl) + state.logvol
    delta_logz = logaddexp(state.logz, logz_remain) - state.logz
    @test delta_logz ≤ 1.0

    chains, state = sample(rng, model, spl; add_live=false, maxiter=3)
    @test state.it == 3

    chains, state = sample(rng, model, spl; add_live=false, maxcall=10)
    @test state.ncall == 10

    chains, state = sample(rng, model, spl; add_live=false, maxlogl=0.2)
    @test state.logl[1] ≥ 0.2
end
