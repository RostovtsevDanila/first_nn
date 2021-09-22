from neuron_network import NeuronNetwork

from math import exp

# def sigmoid(v):
#     return 1 / (1 + exp(-v))


if __name__ == '__main__':
    nnw = NeuronNetwork(
        input_nodes=3,
        hidden_nodes=3,
        output_nodes=3,
        learning_rate=0.5,
    )
    print(nnw.w_input_to_hidden)
    print(nnw.w_hidden_to_output)
    print(nnw.query((1.0, 1.0, 0.0)))
