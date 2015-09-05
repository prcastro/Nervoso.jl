# Cost functions
export quaderror, ceerror

#============================#
#      QUADRATIC ERROR       #
#============================#

"Quadratic error between the output of the network and a target"
quaderror(output, target) = 0.5*norm(target - output)^2

"Derivative of the quadratic error with respect to the outputs"
quaderrorprime(output, target) = (output - target)

#============================#
#    CROSS-ENTROPY ERROR     #
#============================#

"Cross-entropy error between the output of the network and a target"
ceerror(output, target) = -output'log(target)

"Derivative of the cross-entropy error with respect to the outputs"
ceerrorprime(output, target) = -target ./ output
