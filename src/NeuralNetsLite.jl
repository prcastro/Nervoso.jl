module NeuralNetsLite

import Base: show, length, size, endof, getindex, setindex!, start, next, done
export activate, update!, propagate!, train!, sampleerror

"Outer product"
⊗(a,b) = a*b'

"Dictionary associating functions with their derivatives"
derivatives = Dict{Function, Function}()

"Derivative of a function"
der(f::Function) = derivatives[f]

include("activation.jl")
include("feedforward/layer.jl")
include("error.jl")
include("feedforward/network.jl")

end # module NeuralNetsLite
