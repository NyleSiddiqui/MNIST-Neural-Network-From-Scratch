import numpy as np
class Softmax:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dvalues = dvalues

class Sigmoid:
    #Functional, yet NN does not learn
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
        self.inputs = inputs    # Remember input values
        self.output = np.maximum(0, inputs)     # Calculate output values from input

    # Backward pass
    def backward(self, dvalues):
        dvalues = dvalues.copy()  # Since we are modifying original variable, let's make a copy of values first
        dvalues[self.inputs <= 0] = 0
        self.dvalues = dvalues

# helper functions for Sigmoid
def sigmoid(x):
    "Numerically stable sigmoid function. Taken and modifed from Tim Vieira Exp-normalize trick blog post https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/"
    for array in x:
        for element in array:
            if element >= 0:
                z = np.exp(-element)
                x[np.where(x == element)] =  1 / (1 + z)
            else:
                #if element is less than zero then z will be small, denom can't be zero because it's 1+z.
                try:
                    z = np.exp(element)
                except FloatingPointError:
                    z = 0
                x[np.where(x == element)] = z / (1 + z)
    return x

def sigmoid_prime(z):
    return (1 / (1+np.exp(-z))) * (1-(1 / (1+np.exp(-z))))

