function create_mining_context(::Type{SetType}, max_factor_width) where SetType
    [SetType() for _ in 1:Threads.nthreads()],
    [SetType() for _ in 1:Threads.nthreads()],
    [MaxEntContext{SetType}() for _ in 1:Threads.nthreads()],
    [MaxEntContext{SetType}() for _ in 1:(max_factor_width + 1)]
end

"""
    fit(x[, y]; kwargs...) 
    
Discovers a concise set of informative patterns by using a the Bayesian information criteria and maximum entropy modelling.

# Arguments

- `x::Union{BitMatrix, AbstractMatrix{Bool}, Vector{SetType}}`: Input dataset.
- `y::Vector{Integer}`: Class labels that indicate subgroups in `x`.

# Options

- `min_support::Integer`: Require a minimal support of each pattern.
- `max_factor_size::Integer`: Constraint the maximum number of patterns that each factor of the maximum entropy distribution can model. As inference complexity grows exponentially, the `max_factor_size` is bounded by [`MAX_MAXENT_FACTOR_SIZE`](@ref)=12.
- `max_factor_width::Integer`: Constraint the maximum number of singletons that each factor can model.
- `max_expansions::Integer`: Limit the number of search-space node-expansions per iteration. 
- `max_discoveries::Integer`: Terminate the algorithm after `max_discoveries` discoveries.
- `max_seconds::Float64=Inf`: Terminate the algorithm after approximately `max_seconds` seconds.

# Returns

Returns a factorized maximum entropy distribution [`MaxEnt`](@ref) which contains patterns, singletons, and estimated coefficients.
If `y` is specified, this function returns a distribution per group in `x`.

Note: Extract patterns (discoveries) via [`patterns`](@ref) or the per-group patterns via `patterns.`.

# Example

```julia-repl
julia> using Desc: fit, patterns
julia> p = fit(X; min_support = 2)
julia> patterns(p)

julia> ps = fit(X, y)
julia> patterns.(ps)
```

"""
function fit(X; min_support=2, max_factor_size=8, max_factor_width=50, args...)
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    A, B, C, D = create_mining_context(BitSet, max_factor_width)

    n    = size(X, 1)
    L    = Lattice{Candidate{BitSet}}(X, x -> x.support)
    p    = MaxEnt{BitSet,Float64}(map(s -> s.support / n, L.singletons))
    cost = log(n) / 2

    isforbidden(x::Candidate) = isforbidden_ts!(p::MaxEnt, x.set, max_factor_size, max_factor_width, A::Vector{BitSet})
    discover_patterns!(L,
                       x -> if x.support <= min_support || isforbidden(x)
                           0.0
                       else
                           x.support * (log(x.support / n) - log_expectation_ts!(p, x.set, A, B, C)) - cost
                       end,
                       isforbidden,
                       x -> insert_pattern!(p, x.support / n, x.set, max_factor_size, max_factor_width, D); args...)
    p
end

function fit(X, y; min_support=2, max_factor_size=8, max_factor_width=50, args...)
    @assert max_factor_size <= MAX_MAXENT_FACTOR_SIZE

    if y === nothing
        return fit(X; min_support=min_support, max_factor_size=max_factor_size, max_factor_width=max_factor_width, args...)
    end

    A, B, C, D = create_mining_context(BitSet, max_factor_width)

    masks    = [BitSet(findall(==(i), y)) for i in unique(y)]
    k, n     = length(masks), length.(masks)
    L        = Lattice{Candidate{BitSet}}(X, x -> x.support)
    fr(s, j) = intersection_size(s.rows, masks[j]) / n[j]
    Pr       = MaxEnt{BitSet,Float64}
    p        = Pr[Pr(fr.(L.singletons, j)) for j in eachindex(n)]
    cost     = log(size(X, 1)) * k / 2
    icost    = [log(n) / 2 for n in n]

    isforbidden(x) = isforbidden_ts!(p, x.set, max_factor_size, max_factor_width, A)
    score(x) =
        if x.support < min_support || isforbidden(x)
            0.0
        else
            sum(eachindex(p)) do i
                q = intersection_size(x.rows, masks[i])
                q * (log(q / n[i]::Int) - log_expectation_ts!(p[i], x.set, A, B, C))
            end - cost
        end
    report(x) =
        mapreduce(|, eachindex(p)) do i
            q = intersection_size(x.rows, masks[i])
            h = q * (log(q / n[i]::Int) - log_expectation_ts!(p[i], x.set, A, B, C))
            h > icost[i] && insert_pattern!(p[i], q / n[i], x.set, max_factor_size, max_factor_width, D)
        end

    discover_patterns!(L, score, isforbidden, report; args...)

    p
end
