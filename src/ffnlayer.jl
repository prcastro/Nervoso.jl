"""
`type FFNLayer{N}`

Type representing a Neural Network 1-D layer with `N` neurons and 1 bias unit.

### Fields
* `neurons` (Vector{Float64}): Vector with each neuron and a bias unit (index 1)
* `activation` (Function): Activation function
"""
type FFNLayer{N}
    neurons::Vector{Float64}
    activation::Function
    bias::Bool
end

#############################
#        CONSTRUCTORS       #
#############################
FFNLayer(n::Integer) = FFNLayer{n}(vcat([1.0], zeros(n)), tanh, true)
function FFNLayer(n::Integer, bias::Bool)
    if bias
        return FFNLayer{n}(vcat([1.0], zeros(n)), tanh, bias)
    else
        return FFNLayer{n}(zeros(n), tanh, bias)
    end
end

FFNLayer(n::Integer, f::Function) = FFNLayer{n}(vcat([1.0], zeros(n)), f, true)
function FFNLayer(n::Integer, f::Function, bias::Bool)
    if bias
        return FFNLayer{n}(vcat([1.0], zeros(n)), f, bias)
    else
        return FFNLayer{n}(zeros(n), f, bias)
    end
end

############################
#      BASIC FUNCTIONS     #
############################
length{N}(l::FFNLayer{N}) = N
size{N}(l::FFNLayer{N})   = N
endof{N}(l::FFNLayer{N}) = N

function show{N}(io::IO, l::FFNLayer{N})
    print(io, "Feedforward Neural Net Layer: ", N, " neurons")
    if l.bias
        print(io, " + bias unit")
    end
    print(io, ", ", string(l.activation))
end

getindex{N}(l::FFNLayer{N}, i) = l.bias ? l.neurons[i+1] : l.neurons[1]

function setindex!{N}(l::FFNLayer{N}, x, i)
    if l.bias
        l.neurons[i+1] = x
    else
        l.neurons[i]   = x
    end
end

start(l::FFNLayer) = 1
done{N}(l::FFNLayer{N}, s) = s > N
next(l::FFNLayer, s) = (l[s], s+1)

############################
#      OTHER FUNCTIONS     #
############################
function activate(l::FFNLayer)
    if l.bias
        return map(Float64, vcat(l.neurons[1], map(l.activation, l)))
    else
        return map(Float64, map(l.activation, l))
    end
end

update!(l::FFNLayer, x::Vector{Float64}) = l[1:end] = x
