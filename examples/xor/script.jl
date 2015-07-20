# Remember to install HDF5 and JLD packages
using HDF5, JLD, NeuralNetsLite

# Initializing our examples and network
@load "data.jld"
net = FFNNet(2, 2, 1)

# Just checking our in-sample error now to compare it later
println("In-Sample Error before training: ", sampleerror(net, ins, outs))

# Let's train out network 500 times using our examples
train!(net, ins, outs, α=0.5, η=0.1, epochs=500)

# Now we compare our new in-sample error
println("In-Sample Error after training: ", sampleerror(net, ins, outs))
