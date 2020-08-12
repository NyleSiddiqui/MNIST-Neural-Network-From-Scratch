
import random
import sys
sys.path.insert(1, 'C:/Users/nyles/PycharmProjects/MNIST-FeedForwardNerualNetwork/neural-networks-and-deep-learning')
import numpy as np
import mnist_loader
from optimizers import SGD, Adagrad, RMSprop, Adam
from activations import ReLU, Softmax
from network import Layer_Dense
from loss import Loss_CategoricalCrossentropy, Loss




def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    layer1 = Layer_Dense(784, 30)
    layer2 = Layer_Dense(30, 10)
    activation1 = ReLU()
    activation2 = Softmax()
    loss_function = Loss_CategoricalCrossentropy()
    optimizer = SGD()
    train_network(layer1, layer2, activation1, activation2, loss_function, optimizer, 200, training_data)


def train_network(layer1, layer2, activation1, activation2, loss, optimizer, epoch, training_data):
    true = 0
    total = 0
    for epoch in range(epoch):
        random.shuffle(training_data)
        for X, y in training_data[:1000]:
            layer1.forward(X)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            activation2.forward(layer2.output)
            data_loss = loss.forward(activation2.output, y)
            regularization_loss = loss.regularization_loss(layer1) + loss.regularization_loss(layer2)
            total_loss = data_loss + regularization_loss
            predictions = np.argmax(activation2.output, axis=1)
            if predictions == y.argmax():
                true += 1
            total += 1
            accuracy = true / total
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
            print('epoch:', epoch, 'acc:', f'{accuracy:.9f}', 'loss:',
                  f'{total_loss:.3f}', '(data_loss:', f'{data_loss:.3f}', 'reg_loss:',
                  f'{regularization_loss:.3f})', 'lr:',
                  optimizer.current_learning_rate)



if __name__ == "__main__":
    main()