# NeuralNetsLite

[![Build Status](https://travis-ci.org/prcastro/NeuralNetsLite.jl.svg?branch=master)](https://travis-ci.org/prcastro/NeuralNetsLite.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)


This package provides a simple implementation of Feedforward Neural Networks.

The main purpose of this package isn't provide a fast implementation of the algorithms, but rather a general and extensible one, alongside an easy-to-read code.

## Installation

To install this package, simply run:

```julia
Pkg.clone("https://github.com/prcastro/NeuralNetsLite.jl.git")
```

## Basic usage

To use this package, run:

```julia
using NeuralNetsLite
```

### Specifying Datasets

This module expects the examples to be organized in two vectors, one of inputs and one of outputs:

```julia
inputs = Vector{Float64}[
   [1.0, 0.0, 2.0],
   [3.0, 4.0, 2.0],
   [1.0, 1.0, 1.0],
   [5.5, 1.0, 2.0]
]

outputs = Vector{Float64}[
   [1.0, 0.0],
   [0.0, 1.0],
   [1.0, 0.0],
   [1.0, 0.0]
]
```

It's important to notice some things about this: first, these are *vector of vectors*. Each inner vector corresponds to an example:

```julia
# Example one consists of a vector of inputs and a vector of outputs

# Input of example one:
julia> inputs[1]
3-element Array{Float64,1}:
 1.0
 0.0
 2.0

# Output of example one:
julia> outputs[1]
2-element Array{Float64,1}:
 1.0
 0.0
```

Each element of `inputs` is a `Vector{Float64}`, and we specified this with the `Vector{Float64}[...]` syntax. The same is true for the vector `outputs`. Also, all inputs must be of the same size, and all outputs must be of the same size as well.

### Creating the network

Suppose your dataset consists of inputs of size 3, and outputs of size 2 (just as in the example before). You can create a Neural Network compatible with this dataset typing:

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

Notice that you can access the network's layers by indexing the vector `net.layers`. This is the vector of layers of this network and each of it's elements is of the type `FFNNLayer`. A variable of type `FFNNLayer` has a field called `activation`, that specifies the activation function associated with that layer. To change the activation of a layer, we simply change that field.

### Making a prediction

While this network isn't trained yet, we already can try to make it predict the output of an input, by typing:

```julia
propagate!(net, [1.0, 2.0, 1.0])
```

Since the network isn't trained yet, the output will be a 2-element vector containing gibberish.

### Training the network

We can train our network using the dataset we specified earlier:

```julia
train!(net, inputs, outputs)
```

If you need to specify things like the learning rate, momentum rate, batch size and number of epochs, you should check the documentation of the function `train!`.

### Checking the network's performance

We can calculate the mean error committed by the network using the `sampleerror` function:

```julia
sampleerror(net, inputs, outputs)
```

If you are trying to compute the classification error of the network, you can use:

```julia
classerror(net, inputs, outputs)
```

The output will be the number of misclassified point of the dataset.

This works by taking the index of the maximum value of the output of the network, and comparing to the index of the maximum value of the example's output. Therefore, it's natural to use `softmax` activation on the last layer and *one-hot* encoding on the examples' outputs, but it's not mandatory.
