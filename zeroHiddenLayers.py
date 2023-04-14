import numpy as np
import matplotlib.pyplot as plt
from helpers import *
train_path = 'mnist_train.csv'
test_path = 'mnist_test.csv'

class ZeroHiddenLayerNeuralNetwork:
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

        print("Train data shape ")
        print(train_data.shape)

        print("Train labels shape ")
        print(train_labels.shape)

        #The connection weights
        self.weight1 = np.random.randn(self.input.shape[1], train_labels.shape[1])

        print("Weight one shape")
        print(self.weight1.shape)


        #The perceptron biases
        self.biases1 = np.random.randn(self.weight1.shape[1])


    #Push a value through the NN
    def feedforward(self, value, label):

        self.initialValue1 = value.dot(self.weight1) + self.biases1

        self.answer1 = softmax(self.initialValue1)

        self.error = self.answer1 - label
        return self.answer1


    def backPropagation(self, input):

        #Cost function
        cost = (1 / self.batch) * self.error

        print("Cost shape ")
        print(cost.shape)

        print("answers shape")
        print(self.answer1.shape)

        weight1 = np.dot(cost.T, self.answer1).T
        updatedBiases1 = np.sum(cost, axis = 0)

        print("weight1")
        print(weight1.shape)

        print("self.weight1")
        print(self.weight1.shape)

        self.biases1 = self.biases1 - self.learningRate * updatedBiases1
        self.weight1 = self.weight1 - self.learningRate * weight1


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
                # self.backPropagation(self.input[start:end])
                loss += np.mean(self.error**2)

            self.loss.append( loss / (self.input.shape[0] // self.batch))

            #Run the validation in a batch method as well
            for batch in range(self.val_input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.feedforward(self.val_input[start:end], self.val_target[start:end])
                accuracy += np.count_nonzero(np.argmax(self.answer1, axis=1) == np.argmax(self.val_target[start:end],axis=1)) / self.batch

            #Print the accuracy
            self.accuracy.append( accuracy*100 / (self.val_input.shape[0] // self.batch))
            print("Epoch {} Loss: {} Accuracy: {}%".format(epoch+1,self.loss[-1],self.accuracy[-1]))


    #Gets the accuracy of the model after training for each epoch
    def getAccuracy(self):
        return self.accuracy


    #Test the final accuracy of the model
    def test(self, data, labels):
        accuracy = 0
        for batch in range(data.shape[0]//self.batch-1):
            start = batch*self.batch
            end = (batch+1)*self.batch
            self.feedforward(data[start:end], labels[start:end])
            accuracy += np.count_nonzero(np.argmax(self.answer1, axis=1) == np.argmax(labels[start:end],axis=1)) / self.batch

        accuracy = accuracy * 100 / (data.shape[0] // self.batch)
        print("Final Accuracy = ", accuracy)
        return accuracy


