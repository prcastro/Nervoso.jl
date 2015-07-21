module NeuralNetsLite

import Base: show, length, size, endof, getindex, setindex!,
             start, next, done, tanh

export activate, update!, propagate!, train!, meanerror, classerror, softmax

include("utils.jl")
include("feedforward/layer.jl")
include("activation.jl")
include("error.jl")
include("feedforward/network.jl")

end # module NeuralNetsLite
