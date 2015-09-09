VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module Nervoso

import Base: ==, show, length, size, endof, getindex, setindex!,
             start, next, done, tanh

include("utils.jl")
include("feedforward/layer.jl")
include("activation.jl")
include("cost.jl")
include("feedforward/network.jl")
include("feedforward/performance.jl")

function __init__()
    # Put every derivative into a Dict in import-time
    derivatives[tanh]      = tanhprime
    derivatives[softmax]   = softmaxprime
    derivatives[quaderror] = quaderrorprime
    derivatives[ceerror]   = ceerrorprime
end

end # module Nervoso
