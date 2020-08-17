import numpy as np
class Loss:
    # Regularization loss calculation
    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0
        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:  # only calculate when factor greater than 0
            regularization_loss += layer.weight_regularizer_l1 * \
                                   np.sum(np.abs(layer.weights))
            # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                                   np.sum(layer.weights * layer.weights)
            # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:  # only calculate when factor greater than 0
            regularization_loss += layer.bias_regularizer_l1 * \
                                   np.sum(np.abs(layer.biases))
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                                   np.sum(layer.biases * layer.biases)
        return regularization_loss

class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = y_pred.shape[0]

        # Probabilities for target values - only if categorical labels
        # if len(y_true.shape) == 1:
        #     y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log((y_pred[range(samples), y_true.argmax(axis=0)]))

        # Mask values - only for one-hot encoded labels
        if len(y_true.shape) == 2:
            negative_log_likelihoods *= y_true.argmax()
            # negative_log_likelihoods = np.dot(negative_log_likelihoods, y_true.T)

        # Overall loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

    # Backward pass
    def backward(self, dvalues, y_true):
        samples = dvalues.shape[0]
        dvalues = dvalues.copy()  # We need to modify variable directly, make a copy first then
        dvalues[range(samples), y_true.argmax(axis=0)] -= 1
        dvalues = dvalues / samples
        self.dvalues = dvalues