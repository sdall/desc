#!/usr/bin/env -S julia -O3 --threads=auto --check-bounds=no 

using Desc, Comonicon, JSON, GZip, CSV

macro timedv(ex)
    quote
        Base.Experimental.@force_compile
        local stats = Base.gc_num()
        local elapsedtime = Base.time_ns()
        Base.cumulative_compile_timing(true)
        local compile_elapsedtimes = Base.cumulative_compile_time_ns()
        local val = Base.@__tryfinally($(esc(ex)),
                                       (elapsedtime = Base.time_ns() - elapsedtime;
                                        Base.cumulative_compile_timing(false);
                                        compile_elapsedtimes = Base.cumulative_compile_time_ns() .- compile_elapsedtimes))
        local diff = Base.GC_Diff(Base.gc_num(), stats)
        local executiontime = Float64(elapsedtime) - sum(compile_elapsedtimes) - diff.total_time
        (value=val, elapsedtime=elapsedtime / 1e9, gctime=diff.total_time / 1e9, compiletime=first(compile_elapsedtimes) / 1e9,
         recompiletime=last(compile_elapsedtimes) / 1e9, executiontime=executiontime / 1e9)
    end
end

read_sets(fp; offset=0) = [BitSet([offset + parse(Int, s) for s in split(t, " ") if s != ""]) for t in readlines(fp)]
read_sets(f::String) = read_sets((endswith(f, ".gz") ? GZip : Base).open(f))
read_labels(f::String) = [parse(Int, s) for s in readlines((endswith(f, ".gz") ? GZip : Base).open(f))]
function normalize_sets(D)
    I = unique(i for t in D for i in t)
    if sort(I) != collect(eachindex(I))
        tr = Dict(e => i for (i, e) in enumerate(I))
        [BitSet([tr[e] for e in t]) for t in D], Dict(i => e for (e, i) in tr)
    else
        D, nothing
    end
end

"""    
# Introduction

This is a command line interface for the __Desc__ algorithm, which discovers non-redundant pattern-sets using maximum entropy modeling and bic.
_Returns:_ A JSON document containing meta information, a list of (per-group) `patterns`, and the `executiontime` in seconds (see `--measure_time`).
_Example:_ `bin/desc.jl data.tsv labels.tsv --min-support=4 > result.json`
_Measuring Execution Time:_ 📝 As Julia is a compiled language which, at the time of writing, has difficulties with reporting the exact compile-time in a multi-threaded scenario, we include the flag `--measure-time`, to trigger the compiler before measuring and reporting the execution time.  

# Arguments

- `x`: Boolean 01 data matrix as a headerless _.tsv_, or as a sparse list-of-sets [_.dat_], where each row `i` is a space-separated list of indices `j`, such that `X[i, j] != 0`. (optionally gzipped [_.gz_]).".
- `y`: Class labels that partition `x` into gorups as a single headerless integer columns of equal length than x. (optionally gzipped [_.gz_]). 

# Options

- `--min_support`: Require a minimal support of each pattern.
- `--max_discoveries`: Terminate the algorithm after `max_discoveries` discoveries.
- `--max_seconds`: Terminate the algorithm after approximately `max_seconds` seconds.
- `--max_factor_size`: Constraint the maximum number of patterns that each factor of the relaxed maximum entropy distribution can model. As inference complexity grows exponentially, the `max_factor_size` is bounded by `MAX_MAXENT_FACTOR_SIZE=12`.
- `--max_factor_width`: Constraint the maximum number of singletons that each factor can model.

# Flags

- `--measure_time`: Ensure precise compile-time estimates during experiments.
"""
@main function desc(x, y = "";
                    min_support::Int64      = 2,
                    max_discoveries::Int64  = typemax(Int64),
                    max_seconds::Float64    = Inf,
                    measure_time::Bool      = false,
                    max_factor_size::Int64  = 8,
                    max_factor_width::Int64 = 50)
    X, vocab = if any(e -> endswith(x, e), (".dat", ".dat.gz"))
        read_sets(x) |> normalize_sets
    else
        CSV.read(opts["x"], CSV.Tables.matrix; header=0, types=Bool), nothing
    end
    Y = !isnothing(y) && !isempty(y) && isfile(y) ? read_labels(y) : nothing

    if measure_time && Threads.nthreads() > 1
        Desc.desc(X, Y; min_support=min_support, max_seconds=oftype(max_seconds, 0), max_discoveries=max_discoveries,
                  max_factor_size=max_factor_size, max_factor_width=max_factor_width)
    end

    t = @timedv p = Desc.desc(X, Y; min_support=min_support, max_seconds=max_seconds, max_discoveries=max_discoveries,
                              max_factor_size=max_factor_size, max_factor_width=max_factor_width)

    out = isnothing(vocab) ? e -> Set(e) : e -> Set(vocab[i] for i in e)
    S = p isa Vector ? [[out(e) for e in Desc.patterns(q)] for q in p] : [out(e) for e in Desc.patterns(p)]
    Dict("patterns" => S, "executiontime" => t.executiontime, "input" => (x, y)) |> JSON.json |> println
end
