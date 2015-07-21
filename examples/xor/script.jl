# Remember to install HDF5 and JLD packages
using HDF5, JLD, NeuralNetsLite

# Initializing our examples and network
@load "data.jld"
net = FFNNet(2, 2, 1)

# Just checking our in-sample error now to compare it later
println("In-Sample Mean Error before training: ", meanerror(net, ins, outs))
println("In-Sample Classification Error before training: ", classerror(net, ins, outs))

# Let's train out network 500 times using our examples
train!(net, ins, outs, α=0.5, η=0.1, epochs=500, batchsize=1)

# Now we compare our new in-sample error
println("In-Sample Mean Error after training: ", meanerror(net, ins, outs))
println("In-Sample Classification Error before training: ", classerror(net, ins, outs))
