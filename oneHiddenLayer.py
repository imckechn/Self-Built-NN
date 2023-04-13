import numpy as np
import matplotlib.pyplot as plt
from helpers import *


class OneHiddenLayerNeuralNetwork:
    def __init__(self, train_data, train_labels, val_data, val_labels, batch=64, learning_rate=0.01, epochs=50):
        self.input = train_data
        self.target = train_labels
        self.val_input = val_data
        self.val_target = val_labels
        self.batch = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = []  # Loss stored in a list so it can be plotted
        self.accuracy = []  # Accuracy stored in a list so it can be plotted

        # The connection weights
        self.weight1 = np.random.randn(self.input.shape[1], 256)
        self.weight2 = np.random.randn(256, train_labels.shape[1])

        # The perceptron biases
        self.b1 = np.random.randn(256)
        self.b2 = np.random.randn(train_labels.shape[1])


    # Push a value through the NN
    def feedforward(self, value, label):
        self.initial_value1 = value.dot(self.weight1) + self.b1
        self.answer1 = relu(self.initial_value1)
        self.initial_value2 = self.answer1.dot(self.weight2) + self.b2
        self.answer2 = softmax(self.initial_value2)
        self.error = self.answer2 - label
        return self.answer2

    def backprop(self, input):
        # Cost function
        cost = (1 / self.batch) * self.error

        # Finding the weight updates for each layer
        weight2 = np.dot(self.answer1.T, cost)

        # Finding the bias updates for each layer
        db2 = np.sum(cost, axis=0)

        # Finding the error and weight updates for the hidden layer
        hidden_error = np.dot(cost, self.weight2.T) * relu_derivative(self.initial_value1)
        weight1 = np.dot(input.T, hidden_error)

        # Finding the bias update for the hidden layer
        db1 = np.sum(hidden_error, axis=0)

        # Update the perceptron weights
        self.weight2 = self.weight2 - self.learning_rate * weight2
        self.weight1 = self.weight1 - self.learning_rate * weight1

        # Update the layer weights
        self.b2 = self.b2 - self.learning_rate * db2
        self.b1 = self.b1 - self.learning_rate * db1

    #The main algi that performs the train
    def train(self):

        #Run Epoch number of times
        for epoch in range(self.epochs):
            loss = 0
            accuracy = 0

            #Batch training for speed
            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.feedforward(self.input[start:end], self.target[start:end])
                self.backprop(self.input[start:end])
                loss += np.mean(self.error**2)

            self.loss.append( loss / (self.input.shape[0] // self.batch))

            #Run the validation in a batch method as well
            for batch in range(self.val_input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.feedforward(self.val_input[start:end], self.val_target[start:end])
                accuracy += np.count_nonzero(np.argmax(self.answer2, axis=1) == np.argmax(self.val_target[start:end],axis=1)) / self.batch

            #Print the accuracy
            self.accuracy.append( accuracy*100 / (self.val_input.shape[0] // self.batch))
            print("Epoch {} Loss: {} Accuracy: {}%".format(epoch+1,self.loss[-1],self.accuracy[-1]))


    def test(self, data, labels):
        accuracy = 0
        for batch in range(data.shape[0]//self.batch-1):
            start = batch*self.batch
            end = (batch+1)*self.batch
            self.feedforward(data[start:end], labels[start:end])
            accuracy += np.count_nonzero(np.argmax(self.answer2, axis=1) == np.argmax(labels[start:end],axis=1)) / self.batch

        accuracy = accuracy * 100 / (data.shape[0] // self.batch)
        print("Final Accuracy = ", accuracy)
        return accuracy