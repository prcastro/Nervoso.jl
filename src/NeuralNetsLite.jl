module NeuralNetsLite

import Base: show, length, size, endof, getindex, setindex!,
             start, next, done, tanh

export activate, update!, propagate!, train!, meanerror, classerror

include("utils.jl")
include("feedforward/layer.jl")
include("activation.jl")
include("cost.jl")
include("feedforward/network.jl")

end # module NeuralNetsLite
