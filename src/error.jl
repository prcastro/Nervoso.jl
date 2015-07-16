#============================#
#      QUADRATIC ERROR       #
#============================#

"Quadratic error between the output of the network and a example's output"
quaderror(last_layer, example, act::Function) = 0.5*norm(example - act(last_layer))^2

"Derivative of the quadratic error with respect to the values of the last layer"
quaderrorprime(last_layer, example, act::Function) = (example - act(last_layer)) .* map(der(act), last_layer)

# Insert quaderrorprime into derivatives dictionary
derivatives[quaderror] = quaderrorprime
