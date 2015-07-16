# activation functions must be applied to vectors
#  and return a vector, as well as their derivatives

export σ

#============================#
#            TANH            #
#============================#
tanhprime(x) = map(xi -> sech(xi)^2, x)
derivatives[tanh] = tanhprime

#============================#
#          SOFTMAX           #
#============================#
function softmax(x)
    Z = sum(exp(x)) # Normalization
    return [exp(xi)/Z for xi in x]
end

#============================#
#          LOGISTIC          #
#============================#
σ(x) = 1/(1 + e^(-x))
