export FFNNet, propagate, train!

"""
`type FFNNet`

Type representing a Neural Network.

### Fields
* `layers` (`Vector{FFNNLayer}`): Vector containing each layer of the network
* `weights` (`Vector{Matrix{Float64}}`): Vector containing the weight matrices between layers
* `inputsize` (`Int`): Input size accepted by this network
"""
type FFNNet
    layers::Vector{FFNNLayer}
    weights::Vector{Matrix{Float64}}
    inputsize::Int
end

#############################
#        CONSTRUCTORS       #
#############################
"""
`FFNNet(sizes::Int...)`

Construct a network given the input size and the sizes of each layer. By default, the hidden layers have an bias unit and the output layer don't. One the other hand, all the layers have `tanh` as activation function by default.

### Arguments
* `sizes` (`Int...`): Integers specifying the sizes for the network. The first is the input size and the rest is the size of each network layer, from the first one up to the size of the output layer.

### Returns
A Neural Network (`FFNNet{N,I}`)
"""
function FFNNet(sizes::Int...)
    @assert length(sizes) >= 3 "Network must have at least one hidden layer"

    # Create an Array of Neural Network Layers of the right sizes
    # The first size corresponds to the input size of the network
    layers = Array(FFNNLayer, length(sizes) - 1)
    for i in 2:length(sizes) - 1
        layers[i - 1] = FFNNLayer(sizes[i])
    end
    layers[end] = FFNNLayer(sizes[end], bias=false) # Last layer without bias

    return FFNNet(layers, sizes[1])
end

"""
`FFNNet(sizes::Int...)`

Construct a network given its layers and its input size.

### Arguments
* `layers` (`Vector{FFNNLayer}`): A vector with all the layers of the network (in order, with the last one being the output layer).
* `inputsize` (`Int`): Integer specifying the input size of the layer.

### Returns
A Neural Network (`FFNNet`)
"""
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

    return FFNNet(layers, weights, inputsize)
end

############################
#      BASIC FUNCTIONS     #
############################
length(net::FFNNet) = length(net.layers)
size(net::FFNNet) = (net.inputsize, length(net.layers))

function show(io::IO, net::FFNNet)
    L, I = size(net)
    print(io, L, " Layers Feedforward Neural Network:\n  Input Size: ", I)
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
`propagate(net::FFNNet, x::Vector{Float64})`

Propagate an input `x` through the network `net` and return the output

### Arguments
* `net` (`FFNNet`): A neural network that will process the input and give the output
* `x` (`Vector{Float64}`): The input vector. This must be of the same size as the input size of `net`.

### Returns
The output of the network (`Vector{Float64}`). This is simply the activation of the last layer of the network after forwardpropagating the input.
"""
function propagate(net::FFNNet, x::Vector{Float64})
    @assert length(x) == net.inputsize "Network does not support input size $length(x), only $I"

    # Insert bias unit on input and update first layer
    update!(net.layers[1], net.weights[1] * vcat([1.0], x))

    # Update all remaining layers
    for i in 2:length(net)
        update!(net.layers[i], net.weights[i] * activate(net.layers[i-1]))
    end

    # Return the activation of the last layer
    return activate(net.layers[end])
end

function constructbatches(perm::Vector{Int}, batchsize::Int)
    batches = Vector{Int}[]
    while length(perm) > 0
        # Size of this batch
        sizebatch = min(length(perm), batchsize)
        push!(batches, splice!(perm, 1:sizebatch))
    end
    return batches
end

function backpropagate(net::FFNNet,
                       output::Vector{Float64},
                       target::Vector{Float64},
                       cost::Function)

    L = length(net)
    # Vector storing one delta vector for each layer
    δ = Array(Vector{Float64}, L)

    # Compute δ for the last layer
    #   δ^L = ∂E/∂(last.neurons)
    last = net.layers[L]  # Last layer
    δ[L] = (der(last.activation)(last))' * der(cost)(output, target)

    # Find δ of previous layers, backwards
    for l in (L-1):-1:1
       layer = net.layers[l] # Current layer
       W = net.weights[l+1]  # W^(l+1)

       # δ^l = ϕ'(s^l) ⊙ W^(l+1)' δ^(l+1)
       δ[l] = (der(layer.activation)(layer))' * (W'δ[l+1])

       # If there is a bias unit, remove the first
       # element of δ^l, that corresponds to it
       layer.bias && deleteat!(δ[l], 1)
    end

    return δ
end

"""
`train!(net::FFNNet,
        inputs::Vector{Vector{Float64}},
        outputs::Vector{Vector{Float64}};
        α::Real = 0.5,
        η::Real = 0.1,
        epochs::Int = 1,
        batchsize::Int = 1,
        cost::Function = quaderror)`

Train the Neural Network using the examples provided in `inputs` and `outputs`.

### Arguments
* `net` (`FFNNet`): Feedforward Neural Network to be trained.
* `inputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This have K elements, K being the number of examples in this dataset. Each input vector must have I elements, where I is the input size of `net`.
* `outputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This must also have K elements, K being the number of examples in this dataset. Each output vector must have O elements, where O is the size of the output layer of `net`.

### Keyword Arguments
* `α` (`Real`, 0.5 by default): Learning Rate.
* `η` (`Real`, 0.1 by default): Momentum Rate.
* `epochs` (`Int`, 1 by default): Number of iterations of the learning algorithm on this dataset.
* `batchsize` (`Int`, 1 by default): Size of the batch used by the algorithm (1 is simply the default stochastic gradient descent).
* `cost` (`Function`, `quaderror` by default): Cost function to be minimized by the learning algorithm.
"""
function train!(net::FFNNet,
                inputs::Vector{Vector{Float64}},
                outputs::Vector{Vector{Float64}};
                α::Real = 0.5,               # Learning rate
                η::Real = 0.1,               # Momentum rate
                epochs::Int = 1,             # Iterations through entire data
                batchsize::Int = 1,          # Number of examples on each iteration
                cost::Function = quaderror)  # Cost function

    L = length(net)

    # Initialize gradient matrices
    grad      = Matrix{Float64}[zeros(i) for i in net.weights]
    last_grad = Matrix{Float64}[zeros(i) for i in net.weights]

    for τ in 1:epochs
        # Select training order randomly
        perm = randperm(length(inputs))

        # Create batches of the correct size
        batches = constructbatches(perm, batchsize)
        for batch in batches
            # Compute the gradient using this entire batch
            for ex in batch
                input_ex   = vcat([1.0], inputs[ex])     # Example's input with bias
                output_ex  = outputs[ex]                 # Example's output
                output_net = propagate(net, inputs[ex]) # Network's output

                # Find the δs using the backpropagation of this example
                δ = backpropagate(net, output_net, output_ex, cost)

                # Find gradients
                grad[1] += δ[1] ⊗ input_ex # First layer     ∇E = δ^1 ⊗ input
                for l in 2:L               # Other layers    ∇E = δ^l ⊗ x^(l-1)
                    grad[l] += δ[l] ⊗ activate(net.layers[l-1])
                end
            end

            for l in 1:L
                # Update Weights using Momentum Gradient Descent
                #  W^(l) = W^(l) - α∇E - η∇E_old
                net.weights[l] -= α*grad[l] + η*last_grad[l]
                # Save last gradients
                last_grad[l][:] = grad[l][:]
                # Reset gradient component for the next batch
                grad[l][:] = 0.0
            end
        end
    end
end
