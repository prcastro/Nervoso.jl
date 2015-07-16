#============================#
#      QUADRATIC ERROR       #
#============================#

"Quadratic error between the output of the network and a example's output"
quaderror(output, example) = 0.5*norm(output - example)^2

"Derivative of the quadratic error with respect to the values of the last layer"
quaderrorprime(last_layer, example, act::Function) = (act(last_layer) - example) .* map(der(act), last_layer)

# Insert quaderrorprime into derivatives dictionary
derivatives[quaderror] = quaderrorprime
