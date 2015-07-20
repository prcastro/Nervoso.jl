# Remember to install HDF5 and JLD packages
using HDF5, JLD, NeuralNetsLite

# Initializing our examples and network
@load "data.jld"
net = FFNNet(400, 25, 10)

# Just checking our in-sample error now to compare it later
println("In-Sample Error before training: ", sampleerror(net, ins, outs))

# Let's train out network 100 times using our examples
for i in 1:100
    train!(net, ins, outs, α=0.1, η=0.) end

# Now we compare our new in-sample error
println("In-Sample Error after training: ", sampleerror(net, ins, outs))
