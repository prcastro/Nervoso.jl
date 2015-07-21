# Remember to install HDF5 and JLD packages
using HDF5, JLD, NeuralNetsLite

# Initializing our examples and network
@load "data.jld"
net = FFNNet(400, 25, 10)

# Just checking our in-sample error now to compare it later
println("In-Sample Mean Error before training: ", manerror(net, ins, outs))
println("In-Sample Classification Error before training: ", classerror(net, ins, outs))

# Let's train out network 100 times using our examples
train!(net, ins, outs, α=0.1, η=0., epochs=100, batch=1)

# Now we compare our new in-sample error
println("In-Sample Mean Error after training: ", manerror(net, ins, outs))
println("In-Sample Classification Error after training: ", classerror(net, ins, outs))
