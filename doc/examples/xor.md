In this example we will try to learn, usign a neural network, the behavior of a deterministic XOR function. The input is a vector of two bits, and the correct output is 1 if exactly one of them is 1, and 0 otherwise.

To do this, we will use the dataset available in the `examples` folder. So, the first thing to do is load the needed libraries and load the data:

```julia
# Remember to install HDF5 and JLD packages
using HDF5, JLD, NeuralNetsLite

# Loading our examples
@load Pkg.dir("NeuralNetsLite", "examples", "xor", "data.jld")
```

To learn from this dataset we will use a neural network with 2 units in the first layer (input), 2 neurons in the hidden layer and 1 neuron in the output layer:

```julia
net = FFNNet(2, 2, 1)
```

Before training, check the error that our model makes in the dataset:

```julia
println("In-Sample Mean Error before training: ", meanerror(net, ins, outs))
println("In-Sample Classification Error before training: ", classerror(net, ins, outs))
```

Now, we just need to train our network using the dataset as example:

```julia
train!(net, ins, outs, α=0.5, η=0.1, epochs=500, batchsize=1)
```

Here we used a learning rate of `0.5` and momentum rate of `0.1`. We trained the network 500 times with the same dataset, one example at a time (`batchsize=1`).

Now, checkout how the error decreased after training:

```julia
println("In-Sample Mean Error before training: ", meanerror(net, ins, outs))
println("In-Sample Classification Error before training: ", classerror(net, ins, outs))
```
