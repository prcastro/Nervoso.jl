Making a prediction
===================

While this network isn't trained yet, we already can try to make it predict the output of an input, by typing:

```julia
propagate(net, [1.0, 2.0, 1.0])
```

Since the network is initialized with random weights, the output in this case will be a 2-element vector containing gibberish.

Training the network
========================

We can train our network using the dataset we specified earlier:

```julia
train!(net, inputs, outputs)
```

If you need to specify things like the learning rate, momentum rate, number of epochs and batch size, you should check the documentation of the function `train!`.
