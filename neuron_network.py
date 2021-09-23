import datetime
import numpy
from scipy.special import expit


class NeuronNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        input_nodes: count nodes input layout
        hidden_nodes: count nodes hidden layout
        output_nodes: count nodes output layout
        learning_rate: learning rate
        """
        self.activate = lambda x: expit(x)

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        # weights
        self.w_input_to_hidden = numpy.random.normal(
            loc=0.0,
            scale=pow(self.hidden_nodes, -0.5),
            size=(self.hidden_nodes, self.input_nodes),
        )
        self.w_hidden_to_output = numpy.random.normal(
            loc=0.0,
            scale=pow(self.output_nodes, -0.5),
            size=(self.output_nodes, self.hidden_nodes),
        )

    def __int__(self, model: str):
        pass

    def train(self, inputs, targets):
        """

        :return:
        """
        inputs = numpy.array(inputs, ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T

        hidden_inputs = numpy.dot(self.w_input_to_hidden, inputs)
        hidden_outputs = self.activate(hidden_inputs)

        final_inputs = numpy.dot(self.w_hidden_to_output, hidden_outputs)
        final_outputs = self.activate(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w_hidden_to_output.T, output_errors)

        # update values
        self.w_hidden_to_output += self.lr * numpy.dot(
            output_errors * final_outputs * (1. - final_outputs),
            numpy.transpose(hidden_outputs)
        )
        self.w_input_to_hidden += self.lr * numpy.dot(
            hidden_errors * hidden_outputs * (1. - hidden_outputs),
            numpy.transpose(inputs)
        )

    def query(self, inputs):
        """

        :return:
        """
        inputs = numpy.array(inputs, ndmin=2).T

        hidden_inputs = numpy.dot(self.w_input_to_hidden, inputs)
        hidden_outputs = self.activate(hidden_inputs)

        final_inputs = numpy.dot(self.w_hidden_to_output, hidden_outputs)
        final_outputs = self.activate(final_inputs)

        return final_outputs

    def save_model(self, p):
        with open(f"{p}_{datetime.datetime.now()}", "w") as f:
            # f.write(self.w_input_to_hidden, self.w_hidden_to_output)
            pass
