
function step(rng, model, sampler::Nested; kwargs...)
    # Initialize particles
    # us are in unit space, vs are in prior space
    us, vs, logl = init_particles(rng, model, sampler)

    # Find least likely point
    logl_dead, idx_dead = findmin(logl)
    u_dead = us[:, idx_dead]
    v_dead = vs[:, idx_dead]

    # update weight using trapezoidal rule
    logvol = log1mexp(-1 / sampler.nactive)
    logwt = logl_dead + logvol

    # sample a new live point without bounds
    point = rand(rng, eltype(us), sampler.ndims)
    bound = Bounds.fit(Bounds.NoBounds, us)
    proposal = Proposals.Rejection()
    u, v, ll, nc = proposal(rng, v_dead, logl_dead, bound, model)

    us[:, idx_dead] .= u
    vs[:, idx_dead] .= v
    logl[idx_dead] = ll

    ncall = since_update = nc

    # update evidence and information
    logz = logwt
    h = logl_dead - logz
    logzerr = sqrt(h / sampler.nactive)
    logvol -= 1 / sampler.nactive

    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = 1, ncall = ncall, us = us, vs = vs, logl = logl, logl_dead = logl_dead,
             logz = logz, logzerr = logzerr, h = h, logvol = logvol,
             since_update = since_update, has_bounds = false, active_bound = nothing)

    return sample, state
end

function step(rng, model, sampler::Nested, state; kwargs...)
    ## Update bounds
    # expected prior volume per live point, X / nactive with X = exp(-it / nactive)
    pointvol = exp(-state.it / sampler.nactive) / sampler.nactive
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

    # update weight
    logwt = state.logvol + logl_dead

    # update evidence and information
    logz = logaddexp(state.logz, logwt)
    h = (exp(logwt - logz) * logl_dead +
         exp(state.logz - logz) * (state.h + state.logz) - logz)
    logzerr = h ≥ 0 ? sqrt(h / sampler.nactive) : NaN
    logvol = state.logvol - 1 / sampler.nactive

    ## prepare returns
    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = it, ncall = ncall, us = state.us, vs = state.vs, logl = state.logl, logl_dead = logl_dead,
             logz = logz, logzerr = logzerr, h = h, logvol = logvol,
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
    vals = mapreduce(t -> hcat(t.v..., exp(t.logwt - state.logz)), vcat, samples)

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

    vals = mapreduce(t -> hcat(t.v..., exp(t.logwt - state.logz)), vcat, samples)

    if check_wsum
        wsum = sum(vals[:, end])
        err = !iszero(state.logzerr) ? 3 * state.logzerr : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end


    return vals, state
end

## Helpers

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


# add remaining live points to `samples`
function add_live_points(samples, model, sampler, state)
    logvol = -state.it / sampler.nactive - log(sampler.nactive)

    prev_logz = state.logz
    prev_h = state.h

    local logl, logz, h, logzerr 
    N = length(samples)

    @inbounds for (i, idx) in enumerate(eachindex(state.logl))
        # get new point
        u = state.us[:, idx]
        v = state.vs[:, idx]
        logl = state.logl[idx]

        # update sampler
        logwt = logvol + logl
        logz = logaddexp(prev_logz, logwt)
        h = (exp(logwt - logz) * logl +
             exp(prev_logz - logz) * (prev_h + prev_logz) - logz)
        logzerr = sqrt(h / sampler.nactive)

        prev_logz = logz
        prev_h = h

        sample = (u = u, v = v, logwt = logwt, logl = logl)
        save!!(samples, sample, N + i, model, sampler)
    end

    state = (it = state.it + sampler.nactive, us = state.us, vs = state.vs, logl = logl,
            logz = logz, logzerr = logzerr, logvol = logvol,
            since_update = state.since_update, has_bounds = state.has_bounds, active_bound = state.active_bound)
    return samples, state
end
