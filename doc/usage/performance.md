Checking the network's performance
==================================

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
