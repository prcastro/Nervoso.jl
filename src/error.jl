#============================#
#      QUADRATIC ERROR       #
#============================#

"Quadratic error between the output of the network and a example's output"
quaderror(output, example) = 0.5*norm(example - output)^2

"Derivative of the quadratic error with respect to the values of the last layer"
quaderrorprime(last_layer::FFNNLayer, example, act::Function) =
    (activate(last_layer) - example) .* activate(last_layer, der(act))

# Insert quaderrorprime into derivatives dictionary
derivatives[quaderror] = quaderrorprime
