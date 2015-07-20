export FFNNet

"""
`type FFNNet{N,I}`

Type representing a Neural Network with `L` layers with input size `I`.

### Fields
* `layers` (Vector{FFNNLayer}): Vector containing each layer of the network
* `weights` (Vector{Matrix{Float64}}): Vector containing the weight matrices between layers
"""
type FFNNet{L,I}
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
    layers[end] = FFNNLayer(sizes[end], bias=false) # Last layer without bias

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
        update!(net.layers[i], net.weights[i] * activate(net.layers[i-1]))
    end

    # Return the activation of the last layer
    return activate(net.layers[end])
end

function backpropagate{L,I}(net::FFNNet{L,I},
                            output::Vector{Float64},
                            target::Vector{Float64},
                            error::Function)
    # Vector storing one delta vector for each layer
    δ = Array(Vector{Float64}, L)

    # Compute δ for the last layer
    #   δ^L = ∂E/∂(last.neurons)
    last = net.layers[L]  # Last layer
    δ[L] = (der(last.activation)(last))' * der(error)(output, target)

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
`train!(net::FFNNet, inputs, outputs, α, error)`

Train the Neural Network with backpropagation using the examples provided, learning rate `α` and the error function `error`.
"""
function train!{L,I}(net::FFNNet{L,I},
                     inputs::Vector{Vector{Float64}},
                     outputs::Vector{Vector{Float64}};
                     α::Real = 0.5,               # Learning rate
                     η::Real = 0.1,               # Momentum rate
                     epochs::Int = 1,
                     error::Function = quaderror) # Error function

    # Initialize gradient matrices
    grad      = Matrix{Float64}[zeros(i) for i in net.weights]
    last_grad = Matrix{Float64}[zeros(i) for i in net.weights]

    for τ in 1:epochs
    for ex in randperm(length(inputs)) # Select training order randomly
        input_ex   = vcat([1.0], inputs[ex])     # Example's input with bias
        output_ex  = outputs[ex]                 # Example's output
        output_net = propagate!(net, inputs[ex]) # Network's output

        # Find the δs using the backpropagation of this example
        deltas = backpropagate(net, output_net, output_ex, error)

        # Momentum Gradient Descent  W^(L) = W^(L) - α∇E - η∇E

        # Find gradients
        grad[1] = deltas[1] ⊗ input_ex # First layer     ∇E = δ^1 ⊗ input
        for l in 2:L                   # Other layers    ∇E = δ^l ⊗ x^(l-1)
            grad[l] = deltas[l] ⊗ activate(net.layers[l-1])
        end

        # Update Weights using Momentum Gradient Descent
        net.weights -= α * grad + η * last_grad

        # Save last gradients
        last_grad = grad
    end
    end
end

"""
`sampleerror{L,I}(net::FFNNet{L,I},inputs::Vector{Vector{Float64}},
   outputs::Vector{Vector{Float64}}; error::Function = quaderror)`

Mean error of the network in this sample (consisting of `inputs` and `outputs`).
The error is measured using `error` function.
"""
function sampleerror{L,I}(net::FFNNet{L,I},
                          inputs::Vector{Vector{Float64}},
                          outputs::Vector{Vector{Float64}};
                          error::Function = quaderror)

    total_error = 0.0
    for ex in eachindex(inputs)
        total_error += error(propagate!(net, inputs[ex]), outputs[ex])
    end
    return total_error/length(inputs)
end

"""
`classerror{L,I}(net::FFNNet{L,I},inputs,outputs)`

Classification error of the network in this sample (consisting of `inputs` and
`outputs`).
The error is measured counting the ammount of misclassified inputs.
"""
classerror{L,I}(net::FFNNet{L,I},inputs,outputs) =
    sum([indmax(outputs[i]) != indmax(propagate!(net, inputs[i])) for i in 1:length(inputs)])
