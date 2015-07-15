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

function FFNNet(sizes::Int...)
    @assert length(sizes) >= 3 "Network must have 3 or more layers"

    # Create an Array of Neural Network Layers of the right sizes
    # The first size corresponds to the input size of the network
    layers = Array(FFNNLayer, length(sizes) - 1)
    for i in 2:length(sizes) - 1
        layers[i - 1] = FFNNLayer(sizes[i])
    end
    layers[end] = FFNNLayer(sizes[end], false) # Last layer without bias

    return FFNNet(layers, sizes[1])
end

function FFNNet(layers::Vector{FFNNLayer}, inputsize::Int)
    # Create a vector of weight matrices
    weights = Array(Matrix{Float64}, length(layers))

    ɛ(a,b) = sqrt(6)/(sqrt(a + b))

    # The first weight matrix is from the input (including bias)
    #  to the first layer (excluding bias unit)
    eps = ɛ(size(layers[1]), inputsize)
    weights[1] = rand(size(layers[1]), inputsize + 1)*2*eps - eps

    # Matrices from layer i-1 (including bias) to layer i
    for i in 2:length(layers)
        eps = ɛ(size(layers[i]), size(layers[i-1]))
        weights[i] = rand(size(layers[i]), size(layers[i-1]) + 1)
    end

    return FFNNet{length(layers), inputsize}(layers, weights)
end

length{N,I}(net::FFNNet{N,I}) = N

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
                     outputs::Vector{Vector{Float64}};
                     α::Real = 0.05,           # Learning rate
                     error::Function = quaderror) # Error function
    L = length(net)

    for ex in eachindex(inputs)
        input_ex = vcat([1.0], inputs[ex])
        output_ex = outputs[ex]
        output_net = propagate!(net, inputs[ex]) # Forward propagate

        delta = Array(Vector{Float64}, L)

        # Compute δ for the last layer
        #   δ^L = error'(y, ŷ) ⊙ ϕ'(s^L)
        lastlayer = net.layers[L]
        delta[L] = der(error)(output_net, output_ex) .* activate(lastlayer, der(lastlayer.activation))

        # Backpropagate
        for l in (L-1):-1:1
            layer = net.layers[l] # δ^l
            upweights = net.weights[l+1][:,2:end] # w^(l+1) without first column
                                                  # (corresponding to bias unit)
            # δ^l = ϕ'(s^l) ⊙ w^(l+1)'δ^(l+1)
            delta[l] =  activate(layer, der(layer.activation))[2:end] .* (upweights'delta[l+1])
        end

        # Gradient Descent
        net.weights[1] -= α * (delta[1] ⊗ input_ex)
        for l in 2:L
            net.weights[l] -= α * (delta[l] ⊗ net.layers[l-1].neurons)
        end
    end
end

quaderror(output, example) = (norm(output - example))^2
quaderrorprime(output, example) = 2(norm(output - example))
derivatives[quaderror] = quaderrorprime
