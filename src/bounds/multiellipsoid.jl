
"""
    Bounds.MultiEllipsoid([T=Float64], ndims)
    Bounds.MultiEllipsoid(::AbstractVector{Ellipsoid})

Use multiple [`Ellipsoid`](@ref)s in an optimal clustering to bound prior space. This implementation follows the MultiNest implementation outlined in Feroz et al. (2008,2009).[^3][^4] For more details about the bounding algorithm, see the extended help (`??Bounds.MultiEllipsoid`)

[^3]: Feroz and Hobson, 2008, MNRAS 384, 2 ["Multimodal nested sampling: an efficient and robust alternative to Markov Chain Monte Carlo methods for astronomical data analyses"](https://academic.oup.com/mnras/article/384/2/449/1023701)
[^4]: Feroz et al., 2009, MNRAS 398, 4 ["MultiNest: an efficient and robust Bayesian inference tool for cosmology and particle physics"](https://academic.oup.com/mnras/article/398/4/1601/981502)

## Extended help

The multiple-ellipsoidal implementation follows dynesty 2.1.5 (Speagle 2020):

1. Fit a [`Bounds.Ellipsoid`](@ref) to the sample.
2. Perform K-means clustering (here using [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)), started at the endpoints of the major axis of the bounding ellipsoid and run in coordinates scaled by the standard deviation of the sample. This defines two clusters within the sample.
3. If either cluster has fewer than `2ndims` points, consider it ill-defined and end any recursion.
4. Fit [`Bounds.Ellipsoid`](@ref) to each of the clusters assigned in (2), and recurse (1-5) into each.
5. Keep the split if it lowers the total volume `V` by more than a penalty for the extra ellipsoid, `log(V_split) - log(V) < -nparam * log(n) / n`, with `nparam = d (d + 3) / 2` parameters per ellipsoid and `n` points (half a BIC penalty); otherwise return the single ellipsoid.

An ellipsoid fitted to few points covers only part of the region they were drawn from (in 5 dimensions about 70% for 20 points), so splitting into many small clusters cuts off parts of the likelihood contour and biases ln Z upward; the penalty prevents that.

To sample uniformly from the union, an ellipsoid is selected with probability proportional to its volume and a point is drawn from it, which is accepted with probability `1/q`, where `q` is the number of ellipsoids that contain the point.
"""
struct MultiEllipsoid{T} <: AbstractBoundingSpace{T}
    ellipsoids::Vector{Ellipsoid{T}}
    weights::Vector{T}  # volume fractions, cached at the first draw; emptied by `scale!`
end
MultiEllipsoid(ellipsoids::AbstractVector{Ellipsoid{T}}) where {T} = MultiEllipsoid{T}(ellipsoids, T[])
MultiEllipsoid(ndims::Integer) = MultiEllipsoid(Float64, ndims)
MultiEllipsoid(T::Type, ndims::Integer) = MultiEllipsoid([Ellipsoid(T, ndims)])

Base.eltype(::MultiEllipsoid{T}) where {T} = T
Base.broadcastable(me::MultiEllipsoid) = (me,)
Base.length(me::MultiEllipsoid) = length(me.ellipsoids)
Base.ndims(me::MultiEllipsoid) = ndims(me.ellipsoids[1])

volume(me::MultiEllipsoid) = sum(volume, me.ellipsoids)

function scale!(me::MultiEllipsoid, factor)
    scale!.(me.ellipsoids, factor)
    empty!(me.weights)
    return me
end

# fraction of the total volume in each ellipsoid
function volume_weights(me::MultiEllipsoid)
    if isempty(me.weights)
        vols = volume.(me.ellipsoids)
        append!(me.weights, vols ./ sum(vols))
    end
    return me.weights
end

# index drawn with probabilities `w`: the same draw as `rand(rng, Distributions.Categorical(w))`,
# without constructing the distribution
function rand_index(rng::AbstractRNG, w::AbstractVector)
    draw = rand(rng, eltype(w))
    cp = w[1]
    i = 1
    while cp <= draw && i < length(w)
        cp += w[i += 1]
    end
    return i
end

function fit(::Type{<:MultiEllipsoid}, x::AbstractMatrix; pointvol = 0)
    parent = fit(Ellipsoid, x, pointvol = pointvol)
    ells = fit(MultiEllipsoid, x, parent, pointvol = pointvol)
    return MultiEllipsoid(ells)
end

# Recursive splitting of the points `x` bounded by `parent`, following `_bounding_ellipsoids` in
# dynesty 2.1.5: k-means with k = 2, started from the endpoints of the major axis and run in
# coordinates scaled by the standard deviation of all points; clusters need at least 2ndim points;
# a split is kept when it lowers the total volume by more than a BIC-like penalty for the extra
# ellipsoid's ndim (ndim + 3) / 2 parameters.
function fit(::Type{<:MultiEllipsoid}, x::AbstractMatrix, parent::Ellipsoid; pointvol = 0, scale = nothing)
    ndim, npoints = size(x)
    min_size = 2ndim
    npoints < 2min_size && return [parent]

    if scale === nothing
        μ = sum(x, dims = 2) ./ npoints
        scale = vec(sqrt.(sum(abs2, x .- μ, dims = 2) ./ (npoints - 1)))
        scale[iszero.(scale)] .= 1
    end
    p1, p2 = endpoints(parent)
    R = kmeans!(x ./ scale, hcat(p1, p2) ./ scale; maxiter = 10)
    in1 = assignments(R) .== 1
    x1 = x[:, in1]
    x2 = x[:, .!in1]
    (size(x1, 2) < min_size || size(x2, 2) < min_size) && return [parent]

    # Getting bounding ellipsoid for each cluster
    ell1, ell2 = fit.(Ellipsoid, (x1, x2), pointvol = pointvol)

    # Splitting pays off if it lowers the BIC, 2 npoints log(V) + k nparam log(npoints) for k
    # ellipsoids of total volume V (dynesty uses half this penalty, as here).
    nparam = ndim * (ndim + 3) ÷ 2
    log_vol_dec = nparam * log(npoints) / npoints
    out = vcat(fit(MultiEllipsoid, x1, ell1; pointvol = pointvol, scale = scale),
               fit(MultiEllipsoid, x2, ell2; pointvol = pointvol, scale = scale))
    # accept if the first split already shrinks the volume enough, or else the full recursion does
    log(volume(ell1) + volume(ell2)) - log(volume(parent)) < -log_vol_dec && return out
    log(sum(volume, out)) - log(volume(parent)) < -log_vol_dec * (length(out) - 1) && return out

    # Otherwise, return single bounding ellipse
    return [parent]
end

Base.in(x::AbstractVector, me::MultiEllipsoid) = any(ell->x ∈ ell, me.ellipsoids)

# normalized distance to the nearest ellipsoid (≤ 1 inside the union)
distance(me::MultiEllipsoid, x::AbstractVector) = minimum(ell -> distance(ell, x), me.ellipsoids)

function Base.rand(rng::AbstractRNG, me::MultiEllipsoid)
    length(me) == 1 && return rand(rng, me.ellipsoids[1])

    weights = volume_weights(me)
    local x

    while true
        # Select random ellipsoid
        idx = rand_index(rng, weights)
        ell = me.ellipsoids[idx]

        # Select point
        x = rand(rng, ell)

        # How many ellipsoids is the sample in
        n = count(ell -> x ∈ ell, me.ellipsoids)

        # Only accept with probability 1/n
        (n == 1 || rand(rng) < 1 / n) && break
    end

    return x
end

"""
Returns a random live point and one of the ellipsoids, whose axes set the scale of walk and
slice proposals (a MultiEllipsoid has no single set of axes), or `nothing` if the point lies
outside all ellipsoids.

The ellipsoid is chosen at random in proportion to its volume, independently of the point.
Choosing an ellipsoid that contains the point makes the proposal depend on where it starts,
so it is no longer symmetric and the walk does not leave the uniform distribution
invariant (dynesty issue #364).
"""
function rand_live(rng::AbstractRNG, me::MultiEllipsoid, us)
    idx = rand(rng, Base.axes(us, 2))
    u = us[:, idx]
    u ∈ me || return u, nothing
    length(me) == 1 && return u, me.ellipsoids[1]
    return u, me.ellipsoids[rand_index(rng, volume_weights(me))]
end
