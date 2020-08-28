# MNIST-Neural-Network-From-Scratch
A Neural Network built from scratch, made to identify handwritten digits in the MNIST database. Based off the book "Neural Networks from Scratch" by Harrison Kinsley and Daniel Kukiela. I followed along with the book and developed my own network, from scratch (no external machine learning libraries) to read in handwritten digits from the MNIST database, and learn to correctly identify them. The best results so far have occured from using ReLU and Softmax activations and using a Stochastic Gradient Descent optimizer. The different parts of the network are split up into different classes for organizational sake. main.py is the only class you need to run the network. Further improvements are on the way, like fully functional alternative optimizers and activations (implemented, yet encountering underflow errors). Code is not optimized and creating the network requires some manual work, as I was more interested in learning the mathematical concepts behind feed-forward neural networks and creating the project from scratch rather than getting the best result or fully optimizing the network. Further updates may be made in the future to solve these problems, such as initialzing weights and biases more effeciently than a random Gaussian distribution. 
