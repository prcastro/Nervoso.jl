# NeuralNetsLite


---

<a id="method__activate.1" class="lexicon_definition"></a>
#### activate(l::NeuralNetsLite.FFNNLayer) [¶](#method__activate.1)
Activation of a Neural Network layer

*source:*
[NeuralNetsLite/src/feedforward/layer.jl:102](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/layer.jl#L102)

---

<a id="method__ceerror.1" class="lexicon_definition"></a>
#### ceerror(output,  target) [¶](#method__ceerror.1)
Cross-entropy error between the output of the network and a target

*source:*
[NeuralNetsLite/src/cost.jl:19](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/cost.jl#L19)

---

<a id="method__classerror.1" class="lexicon_definition"></a>
#### classerror(net::NeuralNetsLite.FFNNet,  inputs::Array{Array{Float64, 1}, 1},  outputs::Array{Array{Float64, 1}, 1}) [¶](#method__classerror.1)
`classerror(net::FFNNet,
                 inputs::Vector{Vector{Float64}},
                 outputs::Vector{Vector{Float64}})`

Classification error of the network in this sample (consisting of `inputs` and
`outputs`). The error is measured counting the ammount of misclassified inputs.

### Arguments
* `net` (`FFNNet`): Feedforward Neural Network
* `inputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This have K elements, K being the number of examples in this dataset. Each input vector must have I elements, where I is the input size of `net`.
* `outputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This must also have K elements, K being the number of examples in this dataset. Each output vector must have O elements, where O is the size of the output layer of `net`.

### Details
This function assumes that a classification of a vector is the index of the maximum value inside that vector. For example, the vector `[0,1,0]` is classified as 2. This function takes the classification of the output of the network, and compares to the classification of the example's output, counting the number of mismatches.

Therefore, it's natural to use `softmax` activation on the last layer of `net` and *one-hot* encoding on the examples' outputs, but it's not mandatory.


*source:*
[NeuralNetsLite/src/feedforward/performance.jl:48](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/performance.jl#L48)

---

<a id="method__der.1" class="lexicon_definition"></a>
#### der(f::Function) [¶](#method__der.1)
Derivative of a function

*source:*
[NeuralNetsLite/src/utils.jl:10](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/utils.jl#L10)

---

<a id="method__meanerror.1" class="lexicon_definition"></a>
#### meanerror(net::NeuralNetsLite.FFNNet,  inputs::Array{Array{Float64, 1}, 1},  outputs::Array{Array{Float64, 1}, 1}) [¶](#method__meanerror.1)
`meanerror(net::FFNNet,inputs::Vector{Vector{Float64}},
   outputs::Vector{Vector{Float64}}; cost::Function = quaderror)`

Mean error of the network in this sample (consisting of `inputs` and `outputs`). The error is measured using `cost` function.

### Arguments
* `net` (`FFNNet`): Feedforward Neural Network
* `inputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This have K elements, K being the number of examples in this dataset. Each input vector must have I elements, where I is the input size of `net`.
* `outputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This must also have K elements, K being the number of examples in this dataset. Each output vector must have O elements, where O is the size of the output layer of `net`.


### Keyword Arguments
* `cost` (`Function`, `quaderror` by default): Cost function to be used as error measure.


*source:*
[NeuralNetsLite/src/feedforward/performance.jl:18](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/performance.jl#L18)

---

<a id="method__propagate.1" class="lexicon_definition"></a>
#### propagate(net::NeuralNetsLite.FFNNet,  x::Array{Float64, 1}) [¶](#method__propagate.1)
`propagate(net::FFNNet, x::Vector{Float64})`

Propagate an input `x` through the network `net` and return the output

### Arguments
* `net` (`FFNNet`): A neural network that will process the input and give the output
* `x` (`Vector{Float64}`): The input vector. This must be of the same size as the input size of `net`.

### Returns
The output of the network (`Vector{Float64}`). This is simply the activation of the last layer of the network after forwardpropagating the input.


*source:*
[NeuralNetsLite/src/feedforward/network.jl:112](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/network.jl#L112)

---

<a id="method__quaderror.1" class="lexicon_definition"></a>
#### quaderror(output,  target) [¶](#method__quaderror.1)
Quadratic error between the output of the network and a target

*source:*
[NeuralNetsLite/src/cost.jl:9](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/cost.jl#L9)

---

<a id="method__train.1" class="lexicon_definition"></a>
#### train!(net::NeuralNetsLite.FFNNet,  inputs::Array{Array{Float64, 1}, 1},  outputs::Array{Array{Float64, 1}, 1}) [¶](#method__train.1)
`train!(net::FFNNet,
        inputs::Vector{Vector{Float64}},
        outputs::Vector{Vector{Float64}};
        α::Real = 0.5,
        η::Real = 0.1,
        epochs::Int = 1,
        batchsize::Int = 1,
        cost::Function = quaderror)`

Train the Neural Network using the examples provided in `inputs` and `outputs`.

### Arguments
* `net` (`FFNNet`): Feedforward Neural Network to be trained.
* `inputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This have K elements, K being the number of examples in this dataset. Each input vector must have I elements, where I is the input size of `net`.
* `outputs` (`Vector{Vector{Float64}}`): Vector containing input examples (each one is a vector). This must also have K elements, K being the number of examples in this dataset. Each output vector must have O elements, where O is the size of the output layer of `net`.

### Keyword Arguments
* `α` (`Real`, 0.5 by default): Learning Rate.
* `η` (`Real`, 0.1 by default): Momentum Rate.
* `epochs` (`Int`, 1 by default): Number of iterations of the learning algorithm on this dataset.
* `batchsize` (`Int`, 1 by default): Size of the batch used by the algorithm (1 is simply the default stochastic gradient descent).
* `cost` (`Function`, `quaderror` by default): Cost function to be minimized by the learning algorithm.


*source:*
[NeuralNetsLite/src/feedforward/network.jl:191](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/network.jl#L191)

---

<a id="method__update.1" class="lexicon_definition"></a>
#### update!(l::NeuralNetsLite.FFNNLayer,  x::Array{Float64, 1}) [¶](#method__update.1)
Update the internal values of the neurons of a layer 

*source:*
[NeuralNetsLite/src/feedforward/layer.jl:105](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/layer.jl#L105)

---

<a id="type__ffnnlayer.1" class="lexicon_definition"></a>
#### NeuralNetsLite.FFNNLayer [¶](#type__ffnnlayer.1)
`type FFNNLayer`

Type representing a Neural Network 1-D layer with `N` neurons and, eventually, a bias unit.

### Fields
* `neurons` (`Vector{Float64}`): Vector with each neuron and, if `bias = true`, a bias unit (index 1)
* `activation` (`Function`): Activation function
* `bias` (`Bool`): True if there is a bias unit in this layer


*source:*
[NeuralNetsLite/src/feedforward/layer.jl:13](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/layer.jl#L13)

---

<a id="type__ffnnet.1" class="lexicon_definition"></a>
#### NeuralNetsLite.FFNNet [¶](#type__ffnnet.1)
`type FFNNet`

Type representing a Neural Network.

### Fields
* `layers` (`Vector{FFNNLayer}`): Vector containing each layer of the network
* `weights` (`Vector{Matrix{Float64}}`): Vector containing the weight matrices between layers
* `inputsize` (`Int`): Input size accepted by this network


*source:*
[NeuralNetsLite/src/feedforward/network.jl:13](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/network.jl#L13)

---

<a id="global__derivatives.1" class="lexicon_definition"></a>
#### derivatives [¶](#global__derivatives.1)
Dictionary associating functions with their derivatives

*source:*
[NeuralNetsLite/src/utils.jl:7](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/utils.jl#L7)


---

<a id="method__call.1" class="lexicon_definition"></a>
#### call(::Type{NeuralNetsLite.FFNNLayer},  n::Integer) [¶](#method__call.1)
`FFNNLayer(n::Integer; bias::Bool = true)`

Construct a 1-D layer of a Neural Network with `n` neurons and, eventually, a bias unit. The layer has `tanh` as activation function

### Arguments
* `n` (`Int`): Number of neurons in this layer (not counting the eventual bias unit)

### Keyword Arguments
* `bias` (`Bool`, `true` by default): True if there is a bias unit in this layer


*source:*
[NeuralNetsLite/src/feedforward/layer.jl:34](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/layer.jl#L34)

---

<a id="method__call.2" class="lexicon_definition"></a>
#### call(::Type{NeuralNetsLite.FFNNLayer},  n::Integer,  f::Function) [¶](#method__call.2)
`FFNNLayer(n::Integer; bias::Bool = true)`

Construct a 1-D layer of a Neural Network with `n` neurons, `f` as activation function and, eventually, a bias unit.

### Arguments
* `n` (`Int`): Number of neurons in this layer (not counting the eventual bias unit)
* `f` (`Function`): Activation function of this layer

### Keyword Arguments
* `bias` (`Bool`, `true` by default): True if there is a bias unit in this layer


*source:*
[NeuralNetsLite/src/feedforward/layer.jl:54](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/layer.jl#L54)

---

<a id="method__call.3" class="lexicon_definition"></a>
#### call(::Type{NeuralNetsLite.FFNNet},  layers::Array{NeuralNetsLite.FFNNLayer, 1},  inputsize::Int64) [¶](#method__call.3)
`FFNNet(sizes::Int...)`

Construct a network given its layers and its input size.

### Arguments
* `layers` (`Vector{FFNNLayer}`): A vector with all the layers of the network (in order, with the last one being the output layer).
* `inputsize` (`Int`): Integer specifying the input size of the layer.

### Returns
A Neural Network (`FFNNet`)


*source:*
[NeuralNetsLite/src/feedforward/network.jl:59](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/network.jl#L59)

---

<a id="method__call.4" class="lexicon_definition"></a>
#### call(::Type{NeuralNetsLite.FFNNet},  sizes::Int64...) [¶](#method__call.4)
`FFNNet(sizes::Int...)`

Construct a network given the input size and the sizes of each layer. By default, the hidden layers have an bias unit and the output layer don't. One the other hand, all the layers have `tanh` as activation function by default.

### Arguments
* `sizes` (`Int...`): Integers specifying the sizes for the network. The first is the input size and the rest is the size of each network layer, from the first one up to the size of the output layer.

### Returns
A Neural Network (`FFNNet{N,I}`)


*source:*
[NeuralNetsLite/src/feedforward/network.jl:33](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/feedforward/network.jl#L33)

---

<a id="method__ceerrorprime.1" class="lexicon_definition"></a>
#### ceerrorprime(output,  target) [¶](#method__ceerrorprime.1)
Derivative of the cross-entropy error with respect to the outputs

*source:*
[NeuralNetsLite/src/cost.jl:22](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/cost.jl#L22)

---

<a id="method__quaderrorprime.1" class="lexicon_definition"></a>
#### quaderrorprime(output,  target) [¶](#method__quaderrorprime.1)
Derivative of the quadratic error with respect to the outputs

*source:*
[NeuralNetsLite/src/cost.jl:12](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/cost.jl#L12)

---

<a id="method__8855.1" class="lexicon_definition"></a>
#### ⊗(a::Array{Float64, 1},  b::Array{Float64, 1}) [¶](#method__8855.1)
Outer product

*source:*
[NeuralNetsLite/src/utils.jl:4](https://github.com/prcastro/NeuralNetsLite.jl/tree/f80446ab45092c6be32ac2e0789517eca5460628/src/utils.jl#L4)

