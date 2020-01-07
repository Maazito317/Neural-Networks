from __future__ import division
import idx2numpy
import numpy as np
import sys
import pickle
import math
import sys

from Homework5.training import train_input

sys.stdout.flush()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Read input and desired output from files
test_input = idx2numpy.convert_from_file('t10k-images.idx3-ubyte');
test_output = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte');

# Parameters
n = 10000  # max 10,000
HIDDEN_LAYER_NEURONS = 200  ## not needed in test file

# Read TRAINED weights from files
f = open('weights_l2.pckl', 'rb')
weights_l2 = pickle.load(f)
f.close()
f = open('weights_l3.pckl', 'rb')
weights_l3 = pickle.load(f)
f.close()

errors = 0;

for n_sample in range(0, n):

    input_layer = train_input[n_sample].flatten().tolist() + [1]  # append bias of hidden (to input layer)

    post_hidden_layer = np.matrix(weights_l2) * np.matrix(input_layer).T  # multiply by matrix for hidden layer
    post_hidden_layer = post_hidden_layer.flatten().tolist()[0]  # format processing
    post_hidden_function = [sigmoid(x) for x in post_hidden_layer]  # pass through function

    add_bias = post_hidden_function + [1]  # append bias of output (to hidden layer)

    post_output_layer = np.matrix(weights_l3) * np.matrix(add_bias).T  # multiply by matrix for output layer
    post_output_layer = post_output_layer.flatten().tolist()[0]  # format processing
    post_output_function = [sigmoid(x) for x in post_output_layer]  # pass through function

    desired_out = [0] * 10
    desired_out[test_output[n_sample]] = 1

    diff = np.subtract(post_output_function, desired_out)

    if (sum(diff) > 0):
        errors += 1

print('Success rate: ' + str((n - errors) / n) + '%')
