module NeuralNetsLite

import Base: show, length, size, endof, getindex, setindex!, start, next, done
export activate, update!, propagate!, train!, ⊗

"Outer product"
⊗(a,b) = a*b'

derivatives = Dict{Function, Function}()
der(f::Function) = derivatives[f]

include("activation.jl")
include("feedforward/layer.jl")
include("feedforward/network.jl")

end # module NeuralNetsLite
