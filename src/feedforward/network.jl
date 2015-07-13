export FFNNet

"""
`type FFNNet{N,I}`

Type representing a Neural Network with `N` layers with input size `I`.

### Fields
* `layers` (Vector{FFNNLayer}): Vector containing each layer of the network
* `weights` (Vector{Matrix{Float64}}): Vector containing the weight matrices between layers
"""
type FFNNet{N,I}
    layers::Vector{FFNNLayer}
    weights::Vector{Matrix{Float64}}
end

function FFNNet(sizeslayers::NTuple, inputsize::Int)
    @assert length(sizeslayers) >= 3 "Network must have 3 or more layers"

    # Create an Array of Neural Network Layers of the right sizes
    layers = Array(FFNNLayer, length(sizeslayers))
    for i in 1:length(sizeslayers)-1
        layers[i] = FFNNLayer(sizeslayers[i])
    end
    layers[end] = FFNNLayer(sizeslayers[end], false) # Last layer without bias

    return FFNNet(layers, inputsize)
end

function FFNNet(layers::Vector{FFNNLayer}, inputsize::Int)
    # Create a vector of weight matrices
    weights = Array(Matrix{Float64}, length(layers))

    # The first weight matrix is from the input (including bias)
    #  to the first layer (excluding bias unit)
    weights[1] = rand(size(layers[1]), inputsize + 1)

    # Matrices from layer i-1 (including bias) to layer i
    for i in 2:length(layers)
        weights[i] = rand(size(layers[i]), size(layers[i-1]) + 1)
    end

    return FFNNet{length(layers), inputsize}(layers, weights)
end

function show{N,I}(io::IO, net::FFNNet{N,I})
    print(io, N, " Layers Feedforward Neural Network:\n  Input Size: ", I)
    for (i, l) in enumerate(net.layers)
        print(io, "\n  Layer ", i, ": ", size(l), " neurons")
        if l.bias
            print(io, " + bias unit")
        end
        print(io, ", ", string(l.activation))
    end
end

"""
`propagate!(net::FFNNet{N,I}, x::Vector{Float64})`

Propagate an input `x` through the network `net` and return the activation of the last layer
"""
function propagate!{N,I}(net::FFNNet{N,I}, x::Vector{Float64})
    @assert length(x) == I "Network does not support input size $length(x), only $I"

    # Insert bias unit on input and update first layer
    update!(net.layers[1], net.weights[1] * vcat([1.0], x))

    # Update all remaining layers
    for i in 2:N
        update!(net.layers[i], net.weights[i]*activate(net.layers[i-1]))
    end

    # Return the activation of the last layer
    return activate(net.layers[end])
end

"""
`train!(net::FFNNet, inputs, outputs)`

Train the Neural Network with backpropagation using the examples provided.
"""
function train!{N,I}(net::FFNNet{N,I},
                     inputs::Vector{Vector{Float64}},
                     outputs::Vector{Vector{Float64}})

end
