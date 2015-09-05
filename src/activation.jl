# activation functions must be applied to vectors
#  and return a vector, as well as their derivatives

export tanhprime, softmax, softmaxprime, σ

#============================#
#            TANH            #
#============================#
function tanh(l::FFNNLayer)
    if l.bias
        return vcat([1.0], map(tanh, l[1:end]))
    else
        return map(tanh, l[1:end])
    end
end

function tanhprime(l::FFNNLayer)
    diagm(map(xi -> sech(xi)^2, l.neurons))
end

#============================#
#          SOFTMAX           #
#============================#
function softmax(l::FFNNLayer)
    Z = sum(exp(l.neurons)) # Normalization
    return [exp(xi)/Z for xi in l.neurons]
end

function softmax(l::FFNNLayer, i::Int)
    Z = sum(exp(l.neurons)) # Normalization
    return exp(l.neurons[i])/Z
end

function softmaxprime(l::FFNNLayer)
    delta(i,j) = (i==j) ? 1 : 0
    return Float64[softmax(l, i)*(delta(i,j) - softmax(l,j)) for i in 1:length(l.neurons), j in 1:length(l.neurons)]
end
