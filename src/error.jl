quaderror(output, example) = (norm(output - example))^2
quaderrorprime(output, example) = 2(norm(output - example))
derivatives[quaderror] = quaderrorprime
