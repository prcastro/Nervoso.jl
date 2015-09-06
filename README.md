# NeuralNetsLite

[![Build Status](https://travis-ci.org/prcastro/NeuralNetsLite.jl.svg?branch=master)](https://travis-ci.org/prcastro/NeuralNetsLite.jl)
[![Coverage Status](https://coveralls.io/repos/prcastro/NeuralNetsLite.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/prcastro/NeuralNetsLite.jl?branch=master)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)


This package provides a simple implementation of Feedforward Neural Networks.

The main purpose of this package isn't to provide a fast implementation of the algorithms, but rather a general and extensible one, alongside an easy-to-read code. A simple paper explaining the concepts behind this package may be found [here](https://www.dropbox.com/s/yxlyowikizkdrut/NeuralNets_jl_Explanation.pdf?dl=0)

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

### Making a prediction

While this network isn't trained yet, we already can try to make it predict the output of an input, by typing:

```julia
propagate(net, [1.0, 2.0, 1.0])
```

Since the network is initialized with random weights, the output in this case will be a 2-element vector containing gibberish.

### Training the network

We can train our network using the dataset we specified earlier:

```julia
train!(net, inputs, outputs)
```

If you need to specify things like the learning rate, momentum rate, number of epochs and batch size, you should check the documentation of the function `train!`.

### Checking the network's performance

We can calculate the mean error committed by the network using the `meanerror` function:

```julia
meanerror(net, inputs, outputs)
```

If you are trying to compute the classification error of the network instead, you can use:

```julia
classerror(net, inputs, outputs)
```

The output will be the number of misclassified point of the dataset.

This works by taking the index of the maximum value of the output of the network, and comparing to the index of the maximum value of the example's output. Therefore, it's natural to use `softmax` activation on the last layer and *one-hot* encoding on the examples' outputs, but it's not mandatory.

It's also easy to define new functions to assess the error committed by the network on a dataset.

## Extending this library

Users can define new cost and activation functions, given that they preserve the expected interface of these kinds of functions.

### New cost function

To define a new cost function, you should define a function with the following signature:

```julia
function newcost(output::Vector{Real}, target::Vector{Real})
```

This function must return a cost of type `Float64`. Alongside this function you must define it's gradient with respect to the output vector, like this:

```julia
function newcostprime(output::Vector{Real}, target::Vector{Real})
```

This function must return a `Vector{Float64}` with the derivatives of the error with respect to each of the `output`'s coordinates. Check the conceptual PDF more details.

After defining both these functions, you must add `newcost` and `newcostprime` to the derivatives dictionary:

```julia
derivatives[newcost] = newcostprime
```

### New activation function

To define a new activation function, you should define a function with the following signature:

```julia
function newactivation(l::FFNNLayer)
```

This function must return a variable of type `Vector{Float64}` containing the activation of each neuron. Alongside this function you must define a function that differentiate the activation of a layer  with respect to its neurons, like this:

```julia
function newactivationprime(l::FFNNLayer)
```

This function must return the Jacobian Matrix of the activation of the layer with respect to the neurons of the layer. Check the conceptual PDF more details.

After defining both these functions, you must add `newactivation` and `newactivationprime` to the derivatives dictionary:

```julia
derivatives[newactivation] = newactivationprime
```
