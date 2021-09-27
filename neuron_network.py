import datetime
import enum
import json
from dataclasses import dataclass
from nis import match
from typing import List, Type

import numpy
from scipy.special import expit


@dataclass
class NeuronData:
    class Activate(enum.Enum):
        SIGMOID = 1

    activate: Activate
    input_layout: int
    output_layout: int
    hidden_layouts: List[int]
    learning_rate: float

    def __init__(self, activate: Activate, input_layout: int, output_layout: int,
                 hidden_layouts: List[int], learning_rate: float):
        self.activate = activate
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.hidden_layouts = hidden_layouts
        self.learning_rate = learning_rate

    def __str__(self):
        return str(
            f"Activate function: {self.activate}"
            f"Input layout {self.input_layout} nodes"
            f"Hidden layouts: {self.hidden_layouts} nodes"
            f"Output layout: {self.output_layout} nodes"
        )


class NeuronNetwork:
    """"""

    def __init__(self, prop: NeuronData = None, from_file=None):
        if from_file:
            pass
        elif prop:
            self.activate = lambda x: expit(x)
            self.properties = prop

            self.layouts = []
            self.layouts.append(self.properties.input_layout)
            [self.layouts.append(l) for l in self.properties.hidden_layouts]
            self.layouts.append(self.properties.output_layout)

            self.widths = []
            for i in range(len(self.layouts) - 1):
                self.widths.append(
                    numpy.random.normal(
                        loc=0.0,
                        scale=pow(self.layouts[i + 1], -0.5),
                        size=(self.layouts[i + 1], self.layouts[i]),
                    )
                )


            # for i in range(1 + self.properties.hidden_layout_count):
            #     layouts.append(
            #         numpy.random.normal(
            #             loc=0.0,
            #             scale=pow(self.hidden_nodes_count, -0.5),
            #             size=(self.hidden_nodes_count, self.input_nodes_count),
            #         )
            #     )
            #
            # self.layouts = [
            #     numpy.random.normal(
            #         loc=0.0,
            #         scale=pow(self.hidden_nodes_count, -0.5),
            #         size=(self.hidden_nodes_count, self.input_nodes_count),
            #     ),
            #     # ...
            #
            #     numpy.random.normal(
            #         loc=0.0,
            #         scale=pow(self.output_nodes_count, -0.5),
            #         size=(self.output_nodes_count, self.hidden_nodes_count),
            #     )
            # ]
            #
            # # weights
            # self.w_input_to_hidden = numpy.random.normal(
            #     loc=0.0,
            #     scale=pow(self.hidden_nodes_count, -0.5),
            #     size=(self.hidden_nodes_count, self.input_nodes_count),
            # )
            # self.w_hidden_to_output = numpy.random.normal(
            #     loc=0.0,
            #     scale=pow(self.output_nodes_count, -0.5),
            #     size=(self.output_nodes_count, self.hidden_nodes_count),
            # )
        else:
            exit(1)

    def train(self, inputs, targets):
        """

        :return:
        """
        inputs = numpy.array(inputs, ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T
        first_inputs = inputs
        outputs = []
        errors = []

        for w in self.widths:
            inputs = numpy.dot(w, inputs)
            outputs.append(self.activate(inputs))

        for i, o in reversed(list(enumerate(outputs))):
            errors.append(targets - o)
            targets = self.widths[i].T

        w_len = len(self.widths)
        for i in reversed(range(w_len)):
            self.widths[i] += self.properties.learning_rate * numpy.dot(
                errors[w_len - i] * outputs[i] * (1. - outputs[i]),
                numpy.transpose(outputs[int(i if i != 0 else first_inputs)])
            )



        # hidden_inputs = numpy.dot(self.w_input_to_hidden, inputs)
        # hidden_outputs = self.activate(hidden_inputs)
        #
        # final_inputs = numpy.dot(self.w_hidden_to_output, hidden_outputs)
        # final_outputs = self.activate(final_inputs)
        #
        # output_errors = targets - final_outputs
        # hidden_errors = numpy.dot(self.w_hidden_to_output.T, output_errors)
        #
        # # update values
        # self.w_hidden_to_output += self.lr * numpy.dot(
        #     output_errors * final_outputs * (1. - final_outputs),
        #     numpy.transpose(hidden_outputs)
        # )
        # self.w_input_to_hidden += self.lr * numpy.dot(
        #     hidden_errors * hidden_outputs * (1. - hidden_outputs),
        #     numpy.transpose(inputs)
        # )

    def query(self, inputs):
        """

        :return:
        """
        inputs = numpy.array(inputs, ndmin=2).T

        outputs = []
        for w in self.widths:
            inputs = numpy.dot(w, inputs)
            outputs = self.activate(inputs)

        # hidden_inputs = numpy.dot(self.w_input_to_hidden, inputs)
        # hidden_outputs = self.activate(hidden_inputs)
        #
        # final_inputs = numpy.dot(self.w_hidden_to_output, hidden_outputs)
        # final_outputs = self.activate(final_inputs)

        return outputs

    # def save_model(self, folder):
    #     with open(
    #             f"{folder}/{datetime.datetime.now().timestamp()}_{self.input_nodes_count}_{self.hidden_nodes_count}"
    #             f"_{self.output_nodes_count}_{self.lr}.json",
    #             "w",
    #     ) as f:
    #         model_dict = {
    #             "activate": "sigmoid",
    #             "params": {
    #                 "input_nodes": self.input_nodes_count,
    #                 "hidden_nodes": self.hidden_nodes_count,
    #                 "output_nodes": self.output_nodes_count,
    #                 "lr": self.lr,
    #             },
    #             "widths": {
    #                 "w_input_to_hidden": self.w_input_to_hidden.tolist(),
    #                 "w_hidden_to_output": self.w_hidden_to_output.tolist(),
    #             }
    #         }
    #         f.write(json.dumps(model_dict, indent=2))
