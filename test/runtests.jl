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
        @fact FFNNet([layer1, layer2], 10)

        @fact_throws FFNNet(10)
        @fact_throws FFNNet(10,10)
    end

    net1 FFNNet(10,10,10)
    net2 FFNNet(10,10,10,10,10)

    context("Basic informations of a network") do
        @fact length(net1) => 3
        @fact length(net2) => 5
    end

    context("Propagating examples") do
        @fact (typeof(propagate!(net1, ones(10))) == Vector{Float64}) => true
        @fact (typeof(propagate!(net2, ones(10))) == Vector{Float64}) => true
    end

    context("Training") do
    end

    context("Assessing performance") do
    end
end

facts("Activation functions") do
end

facts("Cost functions") do
end
