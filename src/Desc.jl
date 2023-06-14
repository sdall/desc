module Desc

include.(("SetUtils.jl", "MaxEnt.jl", "Lattice.jl", "Discoverer.jl"))

desc(args...; kwargs...) = fit(args...; kwargs...)

export desc, patterns

end # module Desc
