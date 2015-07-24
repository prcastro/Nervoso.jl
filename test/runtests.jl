using NeuralNetsLite
using FactCheck

facts("Layer") do
    context("Build a layer") do
        @fact FFNNLayer(10) => anything
        @fact FFNNLayer(10, softmax, bias = false) => anything

        layer1 = FFNNLayer(10)
        layer2 = FFNNLayer(10, softmax, bias = false)

        context("With Bias") do
            @fact layer1.bias => true
            @fact length(layer1.neurons) => 11
            @fact layer1.activation == tanh => true
        end
        context("Without Bias and different activation function") do
            layer2 = FFNNLayer(10, softmax, bias = false)
            @fact layer2.bias => false
            @fact length(layer2.neurons) => 10
            @fact layer2.activation == softmax => true
        end
    end

    layer1 = FFNNLayer(10)
    layer2 = FFNNLayer(10, softmax, bias = false)

    context("Basic informations of a layer") do
        context("Size and length") do
            @fact size(layer1)   => 10
            @fact length(layer2) => 10
            @fact size(layer1)   => 10
            @fact length(layer2) => 10
        end

        context("Access neurons") do
            @fact layer1[1] => 0
            @fact layer2[1] => 0
            @fact layer1[end] => 0
            @fact layer2[end] => 0
        end

        context("Access bias") do
            @fact layer1[0] => 1.0
            @fact_throws layer2[0]
        end

        context("Iterating on a layer") do
            # Iterate on layer1
            c = 0
            for i in layer1
                @fact i => 0.0
                c += 1
            end
            @fact c => 10

            # Iterate on layer1
            c = 0
            for i in layer2
                @fact i => 0.0
                c += 1
            end
            @fact c => 10
        end
    end

    context("Update and activate a layer") do
        context("Activate") do
            @fact activate(layer1) => vcat([1.0], zeros(10))
            @fact activate(layer2) => ones(10)/10
        end

        context("Update") do
            @fact_throws update(layer1, [1])
            update!(layer1, ones(10))
            @fact layer1.neurons => ones(11)

            @fact_throws update(layer2, [1])
            update!(layer2, ones(10))
            @fact layer2.neurons => ones(10)
        end
    end
end

facts("Network") do
    context("Build a network") do
        @fact FFNNet(10,10,10) => anything
        @fact FFNNet(10,10,10,10,10) => anything

        layer1 = FFNNLayer(10)
        layer2 = FFNNLayer(10, softmax, bias = false)
        @fact FFNNet([layer1, layer2], 10) => anything

        @fact_throws FFNNet(10)
        @fact_throws FFNNet(10,10)
    end

    net1 = FFNNet(2,10,10)
    net2 = FFNNet(2,10,10,10,10)

    context("Basic informations of a network") do
        @fact length(net1) => 2
        @fact length(net2) => 4
    end

    context("Propagating examples") do
        @fact (typeof(propagate(net1, ones(2))) == Vector{Float64}) => true
        @fact (typeof(propagate(net2, ones(2))) == Vector{Float64}) => true
    end

    inputs = Vector{Float64}[
        rand(2),
        rand(2),
        rand(2),
    ]

    outputs = Vector{Float64}[
        rand(10),
        rand(10),
        rand(10),
    ]

    context("Training") do
        @fact train!(net1, inputs, outputs) => nothing
        @fact train!(net2, inputs, outputs) => nothing
    end

    context("Assessing performance") do
        @fact (typeof(meanerror(net1, inputs, outputs)) == Float64) => true
        @fact (typeof(meanerror(net2, inputs, outputs)) == Float64) => true
        @fact (typeof(classerror(net1, inputs, outputs)) == Int) => true
        @fact (typeof(classerror(net2, inputs, outputs)) == Int) => true
    end
end

facts("Activation functions") do
    layer1 = FFNNLayer(10)
    layer2 = FFNNLayer(10, bias = false)

    context("tanh") do
        @fact derivatives[tanh] == tanhprime => true
        @fact der(tanh) == tanhprime => true

        @fact typeof(tanh(layer1)) == Vector{Float64} => true
        @fact length(tanh(layer1)) => length(layer1) + 1
        @fact tanh(layer1) => vcat([1.0], zeros(length(layer1)))
        @fact typeof(der(tanh)(layer1)) == Matrix{Float64} => true
        @fact  size(der(tanh)(layer1)) => (length(layer1) + 1, length(layer1) + 1)

        @fact typeof(tanh(layer1)) == Vector{Float64} => true
        @fact length(softmax(layer2)) => length(layer2)
        @fact tanh(layer2) => zeros(length(layer2))
        @fact typeof(der(tanh)(layer2)) == Matrix{Float64} => true
        @fact  size(der(tanh)(layer2)) => (length(layer2), length(layer2))
    end

    context("softmax") do
        @fact derivatives[softmax] == softmaxprime => true
        @fact der(softmax) == softmaxprime => true

        @fact typeof(softmax(layer1)) == Vector{Float64} => true
        @fact length(softmax(layer1)) => length(layer1) + 1
        @fact typeof(der(softmax)(layer1)) == Matrix{Float64} => true
        @fact  size(der(softmax)(layer1)) => (length(layer1) + 1, length(layer1) + 1)

        @fact typeof(softmax(layer2)) == Vector{Float64} => true
        @fact length(softmax(layer2)) => length(layer2)
        @fact typeof(der(softmax)(layer2)) == Matrix{Float64} => true
        @fact  size(der(softmax)(layer2)) => (length(layer2), length(layer2))
    end

    context("logistic") do
        @pending derivatives[σ] == σprime => true
        @pending der(σ) == σprime => true
        @pending derivatives[logistic] == logisticprime => true
        @pending der(logistic) == logisticprime => true
    end
end

facts("Cost functions") do
end
