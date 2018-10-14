import sys
sys.path.append("./ABAGAIL.jar")

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from func.nn.activation import HyperbolicTangentSigmoid, LogisticSigmoid
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

##########################
# 1. no validation set, since the performance difference between backprop and other 3 algos are large even on training
#    error curve
# 2. network structure (some hyperparamters)
#    - 2 hidden layers
#    - 20 layer size
#    - activation function: sigmoid
#    - loss function: sum squared error / mean squared error ?
#    - stochastic or batch
# 3. the number of iterations

# error history for 3 algos and backprop
#
##########################

INPUT_FILE = os.path.join(".", "php8Mz7BG.csv")

INPUT_LAYER = 5
HIDDEN_LAYER_1 = 20
HIDDEN_LAYER_2 = 20
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 3000


def initialize_instances():
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []

    # Read in the abalone.txt CSV file
    with open(INPUT_FILE, "r") as f:
        # skip the first row
        has_header = csv.Sniffer().has_header(f.read(1024))
        f.seek(0)  # Rewind.
        reader = csv.reader(f)
        if has_header:
            next(reader)  # Skip header row.

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) == 2 else 1))
            instances.append(instance)

    return instances


def train(oa, network, oaName, instances, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)

    err_history = []
    for iteration in xrange(TRAINING_ITERATIONS):    # the number of points tried
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        print "%0.03f" % error
        err_history.append([error])
    return err_history


def main():
    """Run algorithms on the abalone dataset."""
    instances = initialize_instances()
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(instances)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    # oa_names = ["RHC", "SA", "GA"]
    oa_names = ["SA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER,
                                                                      HIDDEN_LAYER_1,
                                                                      HIDDEN_LAYER_2,
                                                                      OUTPUT_LAYER],
                                                                     LogisticSigmoid())
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    # oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .8, nnop[0]))
    # oa.append(StandardGeneticAlgorithm(300, 150, 15, nnop[2]))

    for i, name in enumerate(oa_names):
        start = time.time()
        correct = 0
        incorrect = 0

        err_hist = train(oa[i], networks[i], oa_names[i], instances, measure)
        end = time.time()
        training_time = end - start

        # output error history
        EH_FILE = name+'_3000_0.8.csv'
        with open(EH_FILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(err_hist)

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        start = time.time()
        for instance in instances:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            y_true = instance.getLabel().getContinuous()
            y_prob = networks[i].getOutputValues().get(0)

            if abs(y_true - y_prob) < 0.5:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        results += "\nTraining time: %0.03f seconds" % (training_time,)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

    print results



if __name__ == "__main__":
    main()