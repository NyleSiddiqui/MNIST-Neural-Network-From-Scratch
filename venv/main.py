import random
import sys
import numpy as np
import mnist_loader
from optimizers import SGD, Adagrad, RMSprop, Adam
from activations import ReLU, Softmax, Sigmoid
from network import Layer_Dense, Layer_Dropout
from loss import Loss_CategoricalCrossentropy, Loss

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    layer1 = Layer_Dense(784, 30, weight_regularizer_l2=1e-5, bias_regularizer_l2=1e-5)
    layer2 = Layer_Dense(30, 10)
    dropout1 = Layer_Dropout(0.1) # optional, currently disabled. Used to minimize overfitting
    activation1 = ReLU()
    activation2 = Softmax()
    loss_function = Loss_CategoricalCrossentropy()
    optimizer = Adam()
    run_network(layer1, layer2, activation1, activation2, dropout1, loss_function, optimizer, 200, training_data, len(training_data)) # train network
    run_network(layer1, layer2, activation1, activation2, loss_function, optimizer, 200, test_data, len(test_data)) # test network

def run_network(layer1, layer2, activation1, activation2, dropout, loss, optimizer, epoch, training_data, stop, save=False): # Set save to true if you would like to save the network to disk after training/testing
    true = 0
    for epoch in range(epoch):
        random.shuffle(training_data)
        for X, y in training_data[:stop]:
            layer1.forward(X)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            activation2.forward(layer2.output)
            # print(activation2.output)
            data_loss = loss.forward(activation2.output, y)
            regularization_loss = loss.regularization_loss(layer1) + loss.regularization_loss(layer2)
            total_loss = data_loss + regularization_loss
            predictions = np.argmax(activation2.output, axis=1)
            if predictions == y.argmax(axis=0):
                true += 1
            accuracy = "{0} / {1}".format(true, stop)

            # backward pass
            loss.backward(activation2.output, y)
            activation2.backward(loss.dvalues)
            layer2.backward(activation2.dvalues)
            activation1.backward(layer2.dvalues)
            layer1.backward(activation1.dvalues)
            optimizer.pre_update_params()
            optimizer.update_params(layer1)
            optimizer.update_params(layer2)
            optimizer.post_update_params()

        if not epoch % 10:
            print('epoch:', epoch, 'acc:', accuracy, 'loss:', f'{total_loss:.3f}', '(data_loss:', f'{data_loss:.3f}', 'reg_loss:', f'{regularization_loss:.3f})', 'lr:',  optimizer.current_learning_rate)
        true = 0


    if save:
        saveNN(layer1.weights, layer1.biases, 'l1Weights', 'l1Biases')
        saveNN(layer2.weights, layer2.biases, 'l2Weights', 'l2Biases')
    print("Network is done training")


def mini_batch(data, mini_batch_size):
    # Work in progress. Implement mini batch SGD
    mini_batches = [
        data[k:k+mini_batch_size]
        for k in range(0, len(data), mini_batch_size)]
    return mini_batches

def saveNN(weights, biases, wFilename, bFilename): # Saves network to disk (Only works for two layers)
    np.save('{0}'.format(wFilename), weights)
    np.save('{0}'.format(bFilename), biases)

def loadNN(filename): # Set appropriate layer.weights and/or layer.biases when calling this function to load in previously saved weights/biases from disk
    load = np.load(filename)
    return load

if __name__ == "__main__":
    main()