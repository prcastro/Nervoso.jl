Creating the network
====================

Suppose your dataset consists of inputs of size 3, and outputs of size 2 (just as in the example before). You can create a Neural Network compatible with this dataset by typing:

```julia
# Create a network with input size equal to 3
#  output layer with 2 neurons and a hidden
#  layer with 2 neurons and a bias unit

julia> net = FFNNet(3,2,2)
2 Layers Feedforward Neural Network:
  Input Size: 3
  Layer 1: 2 neurons + bias unit, tanh
  Layer 2: 2 neurons, tanh
```

By default, all layers have `tanh` as it's activation function. To change this, simply type:

```julia
# Change the last layer activation to `softmax`

julia> net.layers[end].activation = softmax
```

Notice that you can access the network's layers by indexing the vector `net.layers`. This is the vector of layers of this network and each of it's elements is of the type `FFNNLayer`.

A variable of type `FFNNLayer` has a field called `activation`, that specifies the activation function associated with that layer. To change the activation of a layer, we simply change that field.
