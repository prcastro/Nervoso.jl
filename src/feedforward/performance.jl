export meanerror, classerror

"""
`meanerror{L,I}(net::FFNNet{L,I},inputs::Vector{Vector{Float64}},
   outputs::Vector{Vector{Float64}}; cost::Function = quaderror)`

Mean error of the network in this sample (consisting of `inputs` and `outputs`).
The error is measured using `cost` function.
"""
function meanerror{L,I}(net::FFNNet{L,I},
                          inputs::Vector{Vector{Float64}},
                          outputs::Vector{Vector{Float64}};
                          cost::Function = quaderror)

    total_error = 0.0
    for ex in eachindex(inputs)
        total_error += cost(propagate!(net, inputs[ex]), outputs[ex])
    end
    return total_error/length(inputs)
end

"""
`classerror{L,I}(net::FFNNet{L,I},
                 inputs::Vector{Vector{Float64}},
                 outputs::Vector{Vector{Float64}})`

Classification error of the network in this sample (consisting of `inputs` and
`outputs`). The error is measured counting the ammount of misclassified inputs.
"""
function classerror{L,I}(net::FFNNet{L,I},
                         inputs::Vector{Vector{Float64}},
                         outputs::Vector{Vector{Float64}})

    sum([indmax(outputs[i]) != indmax(propagate!(net, inputs[i])) for i in 1:length(inputs)])
end
