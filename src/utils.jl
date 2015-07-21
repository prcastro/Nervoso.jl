export derivatives

"Outer product"
⊗(a::Vector{Float64},b::Vector{Float64}) = a*b'

"Dictionary associating functions with their derivatives"
derivatives = Dict{Function, Function}()

"Derivative of a function"
der(f::Function) = derivatives[f]
