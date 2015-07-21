# Cost functions

#============================#
#      QUADRATIC ERROR       #
#============================#

"Quadratic error between the output of the network and a target"
quaderror(output, target) = 0.5*norm(target - output)^2

"Derivative of the quadratic error with respect to the outputs"
quaderrorprime(output, target) = (output - target)

# Insert quaderrorprime into derivatives dictionary
derivatives[quaderror] = quaderrorprime
