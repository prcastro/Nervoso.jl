export FFNNLayer

"""
`type FFNNLayer{N}`

Type representing a Neural Network 1-D layer with `N` neurons and, eventually, a bias unit.

### Fields
* `neurons` (Vector{Float64}): Vector with each neuron and, if `bias = true`, a bias unit (index 1)
* `activation` (Function): Activation function
* `bias` (Bool): True if there is a bias unit in this layer
"""
type FFNNLayer{N}
    neurons::Vector{Float64}
    activation::Function
    bias::Bool
end

#############################
#        CONSTRUCTORS       #
#############################
function FFNNLayer(n::Integer; bias::Bool = true)
    if bias
        return FFNNLayer{n}(vcat([1.0], zeros(n)), tanh, bias)
    else
        return FFNNLayer{n}(zeros(n), tanh, bias)
    end
end

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
activate(l::FFNNLayer) = l.activation(l)

update!(l::FFNNLayer, x::Vector{Float64}) = l[1:end] = x
