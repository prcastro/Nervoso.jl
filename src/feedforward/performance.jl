export meanerror, classerror

"""
`meanerror(net::FFNNet,inputs::Vector{Vector{Float64}},
   outputs::Vector{Vector{Float64}}; cost::Function = quaderror)`

Mean error of the network in this sample (consisting of `inputs` and `outputs`). The error is measured using `cost` function.

### Arguments
* `net` (`FFNNet`): Feedforward Neural Network
* `inputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This have K elements, K being the number of examples in this dataset. Each input vector must have I elements, where I is the input size of `net`.
* `outputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This must also have K elements, K being the number of examples in this dataset. Each output vector must have O elements, where O is the size of the output layer of `net`.


### Keyword Arguments
* `cost` (`Function`, `quaderror` by default): Cost function to be used as error measure.
"""
function meanerror(net::FFNNet,
                   inputs::Vector{Vector{Float64}},
                   outputs::Vector{Vector{Float64}};
                   cost::Function = quaderror)

    total_error = 0.0
    for ex in eachindex(inputs)
        total_error += cost(propagate(net, inputs[ex]), outputs[ex])
    end
    return total_error/length(inputs)
end

"""
`classerror(net::FFNNet,
                 inputs::Vector{Vector{Float64}},
                 outputs::Vector{Vector{Float64}})`

Classification error of the network in this sample (consisting of `inputs` and
`outputs`). The error is measured counting the ammount of misclassified inputs.

### Arguments
* `net` (`FFNNet`): Feedforward Neural Network
* `inputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This have K elements, K being the number of examples in this dataset. Each input vector must have I elements, where I is the input size of `net`.
* `outputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This must also have K elements, K being the number of examples in this dataset. Each output vector must have O elements, where O is the size of the output layer of `net`.

### Details
This function assumes that a classification of a vector is the index of the maximum value inside that vector. For example, the vector `[0,1,0]` is classified as 2. This function takes the classification of the output of the network, and compares to the classification of the example's output, counting the number of mismatches.

Therefore, it's natural to use `softmax` activation on the last layer of `net` and *one-hot* encoding on the examples' outputs, but it's not mandatory.
"""
function classerror(net::FFNNet,
                    inputs::Vector{Vector{Float64}},
                    outputs::Vector{Vector{Float64}})

    return sum([indmax(outputs[i]) != indmax(propagate(net, inputs[i])) for i in 1:length(inputs)])
end
