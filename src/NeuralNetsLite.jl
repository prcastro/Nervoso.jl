VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module NeuralNetsLite

import Base: ==, show, length, size, endof, getindex, setindex!,
             start, next, done, tanh

include("utils.jl")
include("feedforward/layer.jl")
include("activation.jl")
include("cost.jl")
include("feedforward/network.jl")
include("feedforward/performance.jl")

end # module NeuralNetsLite
