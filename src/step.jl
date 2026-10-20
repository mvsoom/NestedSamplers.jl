
function step(rng, model, sampler::Nested; kwargs...)
    # Initialize particles
    # us are in unit space, vs are in prior space
    us, vs, logl = init_particles(rng, model, sampler)

    # Find least likely point
    logl_dead, idx_dead = findmin(logl)
    u_dead = us[:, idx_dead]
    v_dead = vs[:, idx_dead]

    # update weight and evidence; the prior volume shrinks by nactive / (nactive + 1) per
    # iteration in expectation, and L = 0 (log L = -1e300) at X = 1
    dlv, plateau, plateau_logdvol = shrinkage(sampler, logl, logl_dead, 0.0, 0, 0.0)
    logvol = -dlv
    logwt, logz, h = integrate(-1e300, logl_dead, logvol, dlv, -Inf, zero(logl_dead))

    # sample a new live point without bounds
    point = rand(rng, eltype(us), sampler.ndims)
    bound = Bounds.fit(Bounds.NoBounds, us)
    proposal = Proposals.Rejection()
    u, v, ll, nc = proposal(rng, v_dead, logl_dead, bound, model)

    us[:, idx_dead] .= u
    vs[:, idx_dead] .= v
    logl[idx_dead] = ll

    ncall = since_update = nc

    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = 1, ncall = ncall, us = us, vs = vs, logl = logl, logl_dead = logl_dead,
             logz = logz, logzerr = sqrt(max(h, 0) / sampler.nactive), h = h, logvol = logvol,
             plateau = plateau, plateau_logdvol = plateau_logdvol,
             since_update = since_update, has_bounds = false, active_bound = nothing)

    return sample, state
end

function step(rng, model, sampler::Nested, state; kwargs...)
    ## Update bounds
    # expected prior volume per live point, X / nactive
    pointvol = exp(state.logvol) / sampler.nactive
    # check if ready for first update
    if !state.has_bounds && state.ncall > sampler.min_ncall && state.it / state.ncall < sampler.min_eff
        @debug "First update: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = fit_bound(rng, sampler, state.us, pointvol)
        since_update = 0
        has_bounds = true
    # if accepted first update, is it time to update again? `since_update` counts likelihood
    # calls, which grow by more than one per iteration, so compare rather than test divisibility.
    elseif state.has_bounds && state.since_update ≥ sampler.update_interval
        @debug "Updating bounds: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = fit_bound(rng, sampler, state.us, pointvol)
        since_update = 0
        has_bounds = true
    else
        active_bound = state.active_bound
        since_update = state.since_update
        has_bounds = state.has_bounds
    end

    ## Replace least-likely active point
    # Find least likely point
    logl_dead, idx_dead = findmin(state.logl)
    u_dead = state.us[:, idx_dead]
    v_dead = state.vs[:, idx_dead]
    dlv, plateau, plateau_logdvol = shrinkage(sampler, state.logl, logl_dead, state.logvol,
                                              state.plateau, state.plateau_logdvol)

    # sample a new live point using bounds and proposal
    if has_bounds
        u, v, logl, nc, active_bound, refit = propose(rng, sampler.proposal, sampler, active_bound,
                                                      state.us, state.logl, u_dead, logl_dead, model, pointvol)
        refit && (since_update = 0)
    else
        point = rand(rng, eltype(state.us), sampler.ndims)
        bound = Bounds.fit(Bounds.NoBounds, state.us)
        proposal = Proposals.Rejection()
        u, v, logl, nc = proposal(rng, point, logl_dead, bound, model)
    end

    state.us[:, idx_dead] .= u
    state.vs[:, idx_dead] .= v
    state.logl[idx_dead] = logl

    it = state.it + 1
    ncall = state.ncall + nc
    since_update += nc

    # update weight, evidence and information
    logvol = state.logvol - dlv
    logwt, logz, h = integrate(state.logl_dead, logl_dead, logvol, dlv, state.logz, state.h)

    ## prepare returns
    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = it, ncall = ncall, us = state.us, vs = state.vs, logl = state.logl, logl_dead = logl_dead,
             logz = logz, logzerr = sqrt(max(h, 0) / sampler.nactive), h = h, logvol = logvol,
             plateau = plateau, plateau_logdvol = plateau_logdvol,
             since_update = since_update, has_bounds = has_bounds, active_bound = active_bound)

    return sample, state
end

function bundle_samples(samples,
        model::AbstractModel,
        sampler::Nested,
        state,
        ::Type{Chains};
        add_live=true,
        param_names=missing,
        check_wsum=true,
        kwargs...)

    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end
    vals = samples_matrix(samples, state.logz)

    if check_wsum
        wsum = sum(vals[:, end, 1])
        err = !iszero(state.logzerr) ? 3 * state.logzerr : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence=state.logz), state
end

function bundle_samples(samples,
        model::AbstractModel,
        sampler::Nested,
        state,
        ::Type{Array};
        add_live=true,
        check_wsum=true,
        kwargs...)

    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end

    vals = samples_matrix(samples, state.logz)

    if check_wsum
        wsum = sum(vals[:, end])
        err = !iszero(state.logzerr) ? 3 * state.logzerr : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end


    return vals, state
end

## Helpers

# One row per sample: the parameters `v` and the normalized posterior weight. Filled in place;
# `mapreduce(hcat, vcat, samples)` copied the growing matrix for every row, which took most of
# the sampler's time for cheap likelihoods.
function samples_matrix(samples, logz)
    v1 = first(samples).v
    vals = Matrix{promote_type(eltype(v1), typeof(logz))}(undef, length(samples), length(v1) + 1)
    for (i, t) in enumerate(samples)
        for (j, x) in enumerate(t.v)
            vals[i, j] = x
        end
        vals[i, end] = exp(t.logwt - logz)
    end
    return vals
end

xlogx(x) = iszero(x) ? zero(x) : x * log(x)

"""
    shrinkage(sampler, logl, logl_dead, logvol, plateau, plateau_logdvol) -> (dlv, plateau, plateau_logdvol)

Decrease of ln X when the dead point at `logl_dead` is removed from the live points with
log-likelihoods `logl`. Normally `sampler.dlv = log((N + 1) / N)`. When several live points
share the lowest likelihood, a plateau (most often the `-1e300` stand-in for regions of zero
likelihood), each of the `n` tied points removes an equal share X / (N + 1) of the volume, so
that the plateau takes the fraction n / (N + 1) of it rather than 1 - exp(-n / N); as dynesty
2.1. `plateau` counts the tied points still to be removed after this one.
"""
function shrinkage(sampler, logl, logl_dead, logvol, plateau, plateau_logdvol)
    if plateau == 0
        n = count(==(logl_dead), logl)
        n > 1 || return sampler.dlv, 0, plateau_logdvol
        plateau, plateau_logdvol = n, logvol - log(sampler.nactive + 1)
    end
    return -log1p(-exp(plateau_logdvol - logvol)), plateau - 1, plateau_logdvol
end

"""
    integrate(logl_prev, logl, logvol, dlv, logz, h) -> (logwt, logz, h)

One step of the evidence integral with the trapezoid rule, as in dynesty: the shell between
log prior volumes `logvol + dlv` and `logvol`, with log-likelihoods `logl_prev` and `logl` at
its edges, has weight `(L_prev + L) (X_prev - X) / 2`. Returns its log weight and the updated
ln Z and information H.

Follows PR #80 by Miles Lucas. ln Z enters the update of H only through differences (after
Jose Vines), so a dead point at the `-1e300` stand-in for zero likelihood does not inflate H.
The uncertainty of ln Z is Skilling's sqrt(H / nactive); dynesty's running sum of `ΔH dlv`
can turn negative while the final live points are added.
"""
function integrate(logl_prev, logl, logvol, dlv, logz, h)
    logdvol = logvol + log(expm1(dlv)) - log(2)
    logwt = logaddexp(logl_prev, logl) + logdvol
    logz_new = logaddexp(logz, logwt)
    a = exp(logz - logz_new)                # Z_old / Z_new
    b = -expm1(logz - logz_new)             # shell weight / Z_new
    # mean log-likelihood over the shell, with weights normalized explicitly
    m = max(logl_prev, logl)
    w_prev, w = exp(logl_prev - m), exp(logl - m)
    lbar = (w_prev * logl_prev + w * logl) / (w_prev + w)
    h_new = a * h + xlogx(a) + b * (lbar - logz_new)
    return logwt, logz_new, h_new
end

"""
    fit_bound(rng, sampler, us, pointvol)

Fit `sampler.bounds` to the live points `us`, expand it by the bootstrap factor (if
`sampler.bootstrap > 0`), and enlarge its volume by `sampler.enlarge`.
"""
function fit_bound(rng, sampler, us, pointvol)
    bound = Bounds.fit(sampler.bounds, us; pointvol=pointvol)
    if sampler.bootstrap > 0
        expand = Bounds.bootstrap_expand(rng, sampler.bounds, us, sampler.bootstrap; pointvol=pointvol)
        expand > 1 && Bounds.scale!(bound, expand^size(us, 1))
    end
    return Bounds.scale!(bound, sampler.enlarge)
end

"""
    propose(rng, proposal, sampler, bound, us, logl, u_dead, logl_dead, model, pointvol)
        -> (u, v, logl, ncall, bound, refit)

Draw a replacement for the dead point with `proposal` inside `bound`. Returns the new point
in unit and prior space, its log-likelihood, the number of likelihood calls, and the bound,
which is refitted (`refit = true`) when the start point of an MCMC proposal lies outside it.
"""
function propose(rng, proposal::Proposals.Rejection, sampler, bound, us, logl, u_dead, logl_dead, model, pointvol)
    # Rejection sampling draws uniformly from the whole bound: for `MultiEllipsoid`, the union
    # of the ellipsoids, with the 1/q overlap correction in `rand`. It needs no start point.
    u, v, logl, nc = proposal(rng, u_dead, logl_dead, bound, model)
    return u, v, logl, nc, bound, false
end

function propose(rng, proposal, sampler, bound, us, logl, u_dead, logl_dead, model, pointvol)
    # start from a live point strictly above the threshold, not from the dead point (as dynesty)
    above = findall(>(logl_dead), logl)
    point, prop_bound = rand_live(rng, bound, isempty(above) ? us : view(us, :, above))
    refit = isnothing(prop_bound)
    if refit
        # live point not inside active bounds: refit them
        bound = fit_bound(rng, sampler, us, pointvol)
        point, prop_bound = rand_live(rng, bound, point[:, :])
    end
    u, v, logl, nc = proposal(rng, point, logl_dead, prop_bound, model)
    return u, v, logl, nc, bound, refit
end

init_particles(rng, ndims, nactive, model) =
    init_particles(rng, Float64, ndims, nactive, model)

init_particles(rng, model, sampler) =
    init_particles(rng, sampler.ndims, sampler.nactive, model)

# loop and fill arrays, checking validity of points
# will retry 100 times before erroring
function init_particles(rng, T, ndims, nactive, model)
    us = rand(rng, T, ndims, nactive)
    vs_and_logl = mapslices(
        Base.Fix1(prior_transform_and_loglikelihood, model), us;
        dims=1
    )
    vs = mapreduce(first, hcat, vs_and_logl)
    logl = dropdims(map(last, vs_and_logl), dims=1)

    ntries = 1
    while true
        any(isfinite, logl) && break
        rand!(rng, us)
        vs_and_logl .= mapslices(
            Base.Fix1(prior_transform_and_loglikelihood, model), us;
            dims=1
        )
        vs .= mapreduce(first, hcat, vs_and_logl)
        map!(last, logl, vs_and_logl)
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood methods.")
    end

    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    return us, vs, logl
end


# Add the remaining live points to `samples`, in order of increasing likelihood: after the
# last iteration the i-th lowest of the nactive live points bounds an expected prior volume
# X (1 - i / (nactive + 1)). Points left on an unfinished plateau keep its equal shares of
# volume, and the rest split what remains (dynesty).
function add_live_points(samples, model, sampler, state)
    N = sampler.nactive
    k = state.plateau
    share = exp(state.plateau_logdvol - state.logvol)
    rest = k > 0 ? log1p(-k * share) : zero(state.logvol)
    logvols = [i ≤ k ? log1p(-i * share) : rest + log1p(-(i - k) / (N - k + 1)) for i in 1:N]
    nsamples = length(samples)
    logz, h = state.logz, state.h
    logl_prev, logvol_prev = state.logl_dead, state.logvol
    for (i, idx) in enumerate(sortperm(state.logl))
        logl = state.logl[idx]
        logvol = state.logvol + logvols[i]
        logwt, logz, h = integrate(logl_prev, logl, logvol, logvol_prev - logvol, logz, h)
        logl_prev, logvol_prev = logl, logvol
        sample = (u = state.us[:, idx], v = state.vs[:, idx], logwt = logwt, logl = logl)
        samples = save!!(samples, sample, nsamples + i, model, sampler)
    end
    state = merge(state, (it = state.it + N, logl_dead = logl_prev, logz = logz,
                          logzerr = sqrt(max(h, 0) / N), h = h, logvol = logvol_prev, plateau = 0))
    return samples, state
end
