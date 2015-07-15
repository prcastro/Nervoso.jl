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

#############################
#        CONSTRUCTORS       #
#############################
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

############################
#      BASIC FUNCTIONS     #
############################
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

############################
#      OTHER FUNCTIONS     #
############################
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

function backpropagate{L,I}(net::FFNNet{L,I},
                            output_net::Vector{Float64},
                            output_ex::Vector{Float64},
                            error::Function)
    # Vector storing one delta vector for each layer
    deltas = Array(Vector{Float64}, L)

    # Compute δ for the last layer
    #   δ^L = error'(y, ŷ) ⊙ ϕ'(s^L)
    lastlayer = net.layers[L]
    deltas[L] = der(error)(output_net, output_ex) .* activate(lastlayer, der(lastlayer.activation))

    # Find δ of previous layers, backwards
    for l in (L-1):-1:1
       layer = net.layers[l]                 # δ^l
       upweights = net.weights[l+1][:,2:end] # w^(l+1) without first column
                                             #   (that corresponds to the bias unit)

       # δ^l = ϕ'(s^l) ⊙ w^(l+1)' δ^(l+1)
       deltas[l] =  activate(layer, der(layer.activation))[2:end] .* (upweights'deltas[l+1])
    end

    return deltas
end

"""
`train!(net::FFNNet, inputs, outputs)`

Train the Neural Network with backpropagation using the examples provided.
"""
function train!{L,I}(net::FFNNet{L,I},
                     inputs::Vector{Vector{Float64}},
                     outputs::Vector{Vector{Float64}};
                     α::Real = 0.05,              # Learning rate
                     error::Function = quaderror) # Error function

    for ex in eachindex(inputs)
        input_ex = vcat([1.0], inputs[ex]) # Example's input with bias
        output_ex = outputs[ex]            # Example's output

        # Forward propagate the example, updating the neuron values
        #  and obtaining an output
        output_net = propagate!(net, inputs[ex])

        # Find the δs using the backpropagation of this example
        deltas = backpropagate(net, output_net, output_ex, error)

        # Gradient Descent
        net.weights[1] -= α * (deltas[1] ⊗ input_ex)
        for l in 2:L
            net.weights[l] -= α * (deltas[l] ⊗ net.layers[l-1].neurons)
        end
    end
end
