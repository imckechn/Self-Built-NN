import numpy as np
import matplotlib.pyplot as plt
from helpers import *


#The main NN class
class TwoHiddenLayerNeuralNetwork:
    def __init__(self, train_data, train_labels, val_data, val_labels, batch = 64, learningRate = 0.01,  epochs = 50):
        self.input = train_data
        self.target = train_labels
        self.val_input = val_data
        self.val_target = val_labels
        self.batch = batch
        self.epochs = epochs
        self.learningRate = learningRate
        self.loss = []  #Loss stored in a list so it can be plotted
        self.accuracy = []  #Accuracy stored in a list so it can be plotted

        #The connection weights
        self.weight1 = np.random.randn(self.input.shape[1], 256)
        self.weight2 = np.random.randn(self.weight1.shape[1], 128)
        self.weight3 = np.random.randn(self.weight2.shape[1], train_labels.shape[1])

        #The perceptron biases
        self.biases1 = np.random.randn(self.weight1.shape[1])
        self.biases2 = np.random.randn(self.weight2.shape[1])
        self.biases3 = np.random.randn(self.weight3.shape[1])


    #Push a value through the NN
    def feedforward(self, value, label):

        self.initialValue1 = value.dot(self.weight1) + self.biases1
        self.answer1 = relu(self.initialValue1)

        self.initialValue2 = self.answer1.dot(self.weight2) + self.biases2
        self.answer2 = relu(self.initialValue2)

        self.initialValue3 = self.answer2.dot(self.weight3) + self.biases3
        self.answer3 = softmax(self.initialValue3)

        self.error = self.answer3 - label
        return self.answer3


    def backprop(self, input):

        #Cost function
        cost = (1 / self.batch) * self.error

        #Finding the weight updates for each layer
        weight3 = np.dot(cost.T, self.answer2).T

        weight2 = np.dot(
            (np.dot(
                (cost),
                self.weight3.T
            ) * relu_derivative(self.initialValue2)).T,
            self.answer1
        ).T

        weight1 = np.dot(
            (np.dot(
                (np.dot(
                    (cost),
                    self.weight3.T
                ) * relu_derivative(self.initialValue2)),
                self.weight2.T
            ) * relu_derivative(self.answer1)).T,
            input
        ).T

        #Finding the bias updates for each layer
        updatedBiases3 = np.sum(cost, axis = 0)
        updatedBiases2 = np.sum(np.dot((cost),self.weight3.T) * relu_derivative(self.initialValue2),axis = 0)
        updatedBiases1 = np.sum((np.dot(np.dot((cost), self.weight3.T) * relu_derivative(self.initialValue2), self.weight2.T) * relu_derivative(self.answer1)), axis = 0)

        #Update the perceptron weights
        self.weight3 = self.weight3 - self.learningRate * weight3
        self.weight2 = self.weight2 - self.learningRate * weight2
        self.weight1 = self.weight1 - self.learningRate * weight1

        #Update the layer weights
        self.biases3 = self.biases3 - self.learningRate * updatedBiases3
        self.biases2 = self.biases2 - self.learningRate * updatedBiases2
        self.biases1 = self.biases1 - self.learningRate * updatedBiases1


    #The main algi that performs the train
    def train(self):

        #Run Epoch number of times
        for epoch in range(self.epochs):
            loss = 0
            accuracy = 0

            #Batch training for speed
            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch * self.batch
                end = (batch + 1) * self.batch
                self.feedforward(self.input[start:end], self.target[start:end])
                self.backprop(self.input[start:end])
                loss += np.mean(self.error ** 2)

            self.loss.append( loss / (self.input.shape[0] // self.batch))

            #Run the validation in a batch method as well
            for batch in range(self.val_input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.feedforward(self.val_input[start:end], self.val_target[start:end])
                accuracy += np.count_nonzero(np.argmax(self.answer3, axis=1) == np.argmax(self.val_target[start:end],axis=1)) / self.batch

            #Print the accuracy
            self.accuracy.append( accuracy*100 / (self.val_input.shape[0] // self.batch))
            print("Epoch {} Loss: {} Accuracy: {}%".format(epoch+1,self.loss[-1],self.accuracy[-1]))


    def plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")


    def acc_plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.accuracy)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

    #Test the final accuracy of the model
    def test(self, data, labels):
        accuracy = 0
        for batch in range(data.shape[0]//self.batch-1):
            start = batch*self.batch
            end = (batch+1)*self.batch
            self.feedforward(data[start:end], labels[start:end])
            accuracy += np.count_nonzero(np.argmax(self.answer3, axis=1) == np.argmax(labels[start:end],axis=1)) / self.batch

        accuracy = accuracy * 100 / (data.shape[0] // self.batch)
        print("Final Accuracy = ", accuracy)
        return accuracy
