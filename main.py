from typing import List
import numpy

from neuron_network import NeuronNetwork

DATASET_TEST = "./mnist_data/mnist_test.csv"
DATASET_TEST_10 = "./mnist_data/mnist_test_10.csv"
DATASET_TEST_100 = "./mnist_data/mnist_test_100.csv"
DATASET_TRAIN = "./mnist_data/mnist_train.csv"

EPOCH_COUNT = 7


def read_from_csv(p: str) -> ():
    print(f"Reading data from {p} ...")
    with open(p, 'r') as f:
        raw_data = [line.split(',') for line in f.readlines()]
        return (
            numpy.asarray([d[0] for d in raw_data]),  # target value
            numpy.asfarray([f[1:] for f in raw_data]),  # dataset
        )


def prepare_input_data(raw, max_source_val=255., min_target_val=0.001, max_target_val=1.):
    print("Preparing data ...")
    return (raw / max_source_val * (max_target_val - min_target_val)) + min_target_val


if __name__ == '__main__':
    # create and train model
    raw_targets, raw_inputs = read_from_csv(DATASET_TEST_100)
    print(f"RAW DATA\nTargets:\n{raw_targets}\nInputs:\n{raw_inputs}")
    inputs = prepare_input_data(raw_inputs)
    print(f"Prepared inputs:\n{inputs}")

    input_nodes = len(raw_inputs[0])  # image width * height
    hidden_nodes = 300
    output_nodes = 10  # digits [0, 1, ... 8, 9]
    learning_rate = 0.25
    nnw = NeuronNetwork(
        input_nodes,
        hidden_nodes,
        output_nodes,
        learning_rate,
    )
    print(
        f"\nInitiated network parameters:\n"
        f"input_nodes:\t{input_nodes}\n"
        f"hidden_nodes:\t{hidden_nodes}\n"
        f"output_nodes:\t{output_nodes}\n"
        f"learning_rate:\t{learning_rate}\n"
    )
    for epoch in range(EPOCH_COUNT):
        print(f"Start training epoch #{epoch} ...")
        for i, target in enumerate(raw_targets):
            targets = numpy.zeros(output_nodes) + 0.001
            targets[int(target)] = 0.999
            print(f"Epoch: #{epoch}\tStep:{i}\tTarget: {target}\tPrepared targets: {targets}")
            nnw.train(inputs[i], targets)

        print(f"\nTraining epoch#{epoch} done\n")

    nnw.save_model("./models")

    # Testing
    print("\nTesting model ...\n")
    raw_targets, raw_inputs = read_from_csv(DATASET_TEST)
    print(f"RAW DATA\nTargets:\n{raw_targets}\nInputs:\n{raw_inputs}")
    inputs = prepare_input_data(raw_inputs)
    print(f"Prepared inputs:\n{inputs}")

    scores = []

    for i, target in enumerate(raw_targets):
        result = nnw.query(inputs[i])
        result_label = numpy.argmax(result)
        scores.append(1) if int(result_label) == int(target) else scores.append(0)

    print(f"\nModel efficient: {sum(scores) / len(scores) * 100}%")
