module NeuralNetsLite

import Base: show, length, size, endof, getindex, setindex!, start, next, done, tanh
export activate, update!, propagate!, train!, sampleerror, classerror

"Outer product"
⊗(a::Vector{Float64},b::Vector{Float64}) = a*b'

"Dictionary associating functions with their derivatives"
derivatives = Dict{Function, Function}()

"Derivative of a function"
der(f::Function) = derivatives[f]

include("feedforward/layer.jl")
include("activation.jl")
include("error.jl")
include("feedforward/network.jl")

end # module NeuralNetsLite
