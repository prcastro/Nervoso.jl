Extending this library
======================

Users can define new cost and activation functions, given that they preserve the expected interface of these kinds of functions.

New cost function
-----------------

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

New activation function
-----------------------

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
