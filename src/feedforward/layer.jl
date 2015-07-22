export FFNNLayer, activate, update!

"""
`type FFNNLayer{N}`

Type representing a Neural Network 1-D layer with `N` neurons and, eventually, a bias unit.

### Fields
* `neurons` (`Vector{Float64}`): Vector with each neuron and, if `bias = true`, a bias unit (index 1)
* `activation` (`Function`): Activation function
* `bias` (`Bool`): True if there is a bias unit in this layer
"""
type FFNNLayer{N}
    neurons::Vector{Float64}
    activation::Function
    bias::Bool
end

#############################
#        CONSTRUCTORS       #
#############################

"""
`FFNNLayer(n::Integer; bias::Bool = true)`

Construct a 1-D layer of a Neural Network with `n` neurons and, eventually, a bias unit. The layer has `tanh` as activation function

### Arguments
* `n` (`Int`): Number of neurons in this layer (not counting the eventual bias unit)

### Keyword Arguments
* `bias` (`Bool`, `true` by default): True if there is a bias unit in this layer
"""
function FFNNLayer(n::Integer; bias::Bool = true)
    if bias
        return FFNNLayer{n}(vcat([1.0], zeros(n)), tanh, bias)
    else
        return FFNNLayer{n}(zeros(n), tanh, bias)
    end
end

"""
`FFNNLayer(n::Integer; bias::Bool = true)`

Construct a 1-D layer of a Neural Network with `n` neurons, `f` as activation function and, eventually, a bias unit.

### Arguments
* `n` (`Int`): Number of neurons in this layer (not counting the eventual bias unit)
* `f` (`Function`): Activation function of this layer

### Keyword Arguments
* `bias` (`Bool`, `true` by default): True if there is a bias unit in this layer
"""
function FFNNLayer(n::Integer, f::Function; bias::Bool = true)
    if bias
        return FFNNLayer{n}(vcat([1.0], zeros(n)), f, bias)
    else
        return FFNNLayer{n}(zeros(n), f, bias)
    end
end

############################
#      BASIC FUNCTIONS     #
############################
length{N}(l::FFNNLayer{N}) = N
size{N}(l::FFNNLayer{N})   = N
endof{N}(l::FFNNLayer{N})  = N

function show{N}(io::IO, l::FFNNLayer{N})
    print(io, "Feedforward Neural Net Layer: ", N, " neurons")
    if l.bias
        print(io, " + bias unit")
    end
    print(io, ", ", string(l.activation))
end

getindex{N}(l::FFNNLayer{N}, i) = l.bias ? l.neurons[i+1] : l.neurons[i]

function setindex!{N}(l::FFNNLayer{N}, x, i)
    if l.bias
        l.neurons[i+1] = x
    else
        l.neurons[i]   = x
    end
end

start(l::FFNNLayer) = 1
done{N}(l::FFNNLayer{N}, s) = s > N
next(l::FFNNLayer, s) = (l[s], s+1)

############################
#      OTHER FUNCTIONS     #
############################
"Activation of a Neural Network layer"
activate(l::FFNNLayer) = l.activation(l)

"Update the internal values of the neurons of a layer "
update!(l::FFNNLayer, x::Vector{Float64}) = l[1:end] = x
