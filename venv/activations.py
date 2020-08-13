import numpy as np
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
        self.output = Sigmoid.sigmoid(inputs)

    def backward(self, dvalues):
        dvalues = dvalues.copy()
        dvalues = Sigmoid.sigmoid(dvalues) * (1 - Sigmoid.sigmoid(dvalues))
        self.dvalues = dvalues
        
        
    def sigmoid(z):
        try:
            a = np.float64(np.exp(-z))
            return 1.0 / (1.0 + a)
        except FloatingPointError:
            return 1



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