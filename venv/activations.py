import numpy as np
import scipy as sci
from scipy.special import logsumexp
np.seterr(all='raise')

class Softmax:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                             keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dvalues = dvalues

class Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(inputs)

    def backward(self, dvalues):
        dvalues = dvalues.copy()
        dvalues = sigmoid_prime(dvalues)
        self.dvalues = dvalues

class ReLU:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        dvalues = dvalues.copy()  # Since we need to modify original variable, let;s make a copy of values first
        dvalues[self.inputs <= 0] = 0  # Zero gradient where input values were negative
        self.dvalues = dvalues





# helper functions
def sigmoid(x):
    print("pre: {0}".format(x))
    "Numerically stable sigmoid function."
    for array in x:
        for element in array:
            if element >= 0:
                z = np.exp(-element)
                x[np.where(x == element)] =  1 / (1 + z)
            else:
                # if x is less than zero then z will be small, denom can't be
                # zero because it's 1+z.
                try:
                    z = np.exp(element)
                except FloatingPointError:
                    z = 0
                x[np.where(x == element)] = z / (1 + z)
    print("post: {0}".format(x))
    return x


def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

