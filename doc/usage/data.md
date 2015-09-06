Specifying Datasets
===================

This module expects the examples to be organized in two vectors, one of inputs and one of outputs:

```julia
inputs = Vector{Float64}[
   [1.0, 0.0, 2.0],
   [3.0, 4.0, 2.0],
   [1.0, 1.0, 1.0],
   [5.5, 1.0, 2.0]
]

outputs = Vector{Float64}[
   [1.0, 0.0],
   [0.0, 1.0],
   [1.0, 0.0],
   [1.0, 0.0]
]
```

It's important to notice some things about this: first, these are *vector of vectors*. Each inner vector corresponds to an example:

```julia
# Example one consists of a vector of inputs and a vector of outputs

# Input of example one:
julia> inputs[1]
3-element Array{Float64,1}:
 1.0
 0.0
 2.0

# Output of example one:
julia> outputs[1]
2-element Array{Float64,1}:
 1.0
 0.0
```

Each element of `inputs` is a `Vector{Float64}`, and we specified this with the `Vector{Float64}[...]` syntax. The same is true for the vector `outputs`. Also, all inputs must be of the same size, and all outputs must be of the same size as well.
