module NeuralNetsLite

import Base: show, length, size, endof, getindex, setindex!, start, next, done
export activate, update!, propagate!, train!

include("activation.jl")
include("feedforward/net.jl")
include("feedforward/layer.jl")

end # module NeuralNetsLite
