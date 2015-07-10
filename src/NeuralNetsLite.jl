module NeuralNetsLite

import Base: show, length, size, endof, getindex, setindex!, start, next, done
export activate, update!, propagate!, train!, σ

include("actfuns.jl")
include("feedforward.jl")

end # module NeuralNetsLite
