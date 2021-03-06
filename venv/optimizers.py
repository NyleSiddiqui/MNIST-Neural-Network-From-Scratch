import numpy as np
# class SGD:
#     def __init__(self, learning_rate=.01, decay=0):
#         self.learning_rate = learning_rate
#         self.current_learning_rate = learning_rate
#         self.decay = decay
#         self.iterations = 0
#
#     # Call once before any parameter updates
#     def pre_update_params(self):
#         if self.decay:
#             self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))
#
#     # Update parameters
#     def update_params(self, layer):
#         weight_updates = -self.current_learning_rate * layer.dweights
#         bias_updates = -self.current_learning_rate * layer.dbiases
#         layer.weights += weight_updates
#         layer.biases += bias_updates
#
#     # Call once after any parameter updates
#     def post_update_params(self):
#         self.iterations += 1







class SGD:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=.01, decay=4e-12, momentum=0.1, nesterov=True):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.nesterov = nesterov
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))
        # Update parameters
    def update_params(self, layer):
        # If layer does not contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

    # If we use momentum
        if self.momentum:
            # Build weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

            # Apply Nesterov as well?
            if self.nesterov:
                weight_updates = self.momentum * weight_updates - self.current_learning_rate * layer.dweights
                bias_updates = self.momentum * bias_updates - self.current_learning_rate * layer.dbiases

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights with updates which are either vanilla, momentum or momentum+nesterov updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class RMSprop:
    # Non-functional, but I thought I might as well leave it in. Despite ReLU, underflow errors still occur. Possible fix in the future
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.1, decay=1e-10, epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
    (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

    # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 -
                                                          self.rho) * layer.dbiases ** 2
        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Adam:
    # Non-functional, but I thought I might as well leave it in. Despite ReLU, underflow errors still occur. Possible fix in the future
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=.001, decay=4e-8, epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                                         (1. / (1. + (self.decay * self.iterations)))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = (self.beta_1 * layer.weight_momentums)  + ((1 - self.beta_1) * layer.dweights)
        layer.bias_momentums = (self.beta_1 * layer.bias_momentums) + ((1 - self.beta_1) * layer.dbiases)

        # Update cache with squared current gradients
        layer.weight_cache = (self.beta_2 * layer.weight_cache) + ((1 - self.beta_2) * layer.dweights ** 2)
        layer.bias_cache = (self.beta_2 * layer.bias_cache) + ((1 - self.beta_2) * layer.dbiases ** 2)

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - (self.beta_1 ** (self.iterations + 1)))  # self.iteration is 0 at first pass anD we need to start with 1 here
        bias_momentums_corrected = layer.bias_momentums / (1 - (self.beta_1 ** (self.iterations + 1)))

        # Get corrected bias
        weight_cache_corrected = layer.weight_cache / (1 - (self.beta_2 ** (self.iterations + 1)))
        bias_cache_corrected = layer.bias_cache / (1 - (self.beta_2 ** (self.iterations + 1)))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights = layer.weights -  ((self.current_learning_rate * weight_momentums_corrected) / (np.sqrt(weight_cache_corrected) + self.epsilon))
        layer.biases = layer.biases -  ((self.current_learning_rate * bias_momentums_corrected) / (np.sqrt(bias_cache_corrected) + self.epsilon))

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Adagrad:
    # Non-functional, but I thought I might as well leave it in. Despite ReLU, underflow errors still occur. Possible fix in the future
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=.001, decay=4e-8, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))
    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            # Update cache with squared current gradients
            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2
            # Vanilla SGD parameter update + normalization with square rooted cache
            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1