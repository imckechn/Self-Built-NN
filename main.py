import pandas as pd
import random
import math
import numpy as np
import time

def importData(sources):
    data = []

    for source in sources:
        dat = pd.read_csv(source)
        data += dat.values.tolist()

    return data

def trainValTestSplit(data, trainSize, valSize):

    train = data[:int(trainSize*len(data))]
    val = data[int(trainSize*len(data)):int((trainSize+valSize)*len(data))]
    test = data[int((trainSize+valSize)*len(data)):]

    return train, val, test

def getLabels(data):
    labels = []
    mnist = []
    for row in data:
        label = []
        for i in range(10):
            if i == row[0]:
                label.append(1)
            else:
                label.append(0)

        labels.append(label)
        mnist.append(row[1:])

    return labels, mnist


class Connection:
    def __init__(self, previousPerceptron, nextPerceptron, weight = 1):
        self.weight = random.random()
        self.previousPerceptron = previousPerceptron
        self.nextPerceptron = nextPerceptron

    def updateWeight(self, delta):
        self.weight += delta

    def getWeight(self):
        return self.weight

    def getPreviousPerceptron(self):
        return self.previousPerceptron.getOutput()

class Perceptron:

    #Create the perceptron, only takes in it's bias
    def __init__(self, isOutputLayer, bias = 1):
        self.bias = bias
        self.backwardsConnections = []
        self.forwardsConnections = []
        self.output = 0
        self.isOutputLayer = isOutputLayer


    #This method is used to add a forward connection to the perceptron
    def addForwardsConnection(self, connection):
        self.forwardsConnections.append(connection)


    #This takes in the layer before it and creates connections to all of the perceptrons in that layer
    # The connections are made of connection objects which just contain the node before it, the node after it, and the weight
    # The previous layer is a list of perceptrons
    def makeConnections(self, previousLayer):
        for perceptron in previousLayer:
            connection = Connection(perceptron, self)
            perceptron.addForwardsConnection(connection)

            self.backwardsConnections.append(connection)


    #This method returns the output of the perceptron
    def getOutput(self):
        # return self.activationFunction(self.output)
        return self.output


    #The activation function is a sigmoid function
    def activationFunction(self, value):
        print(value)
        return 1/(1+math.exp(-value))


    #This is the function that is called to calculate the output of the perceptron
    #Loops through all the back connections it has and gets the output of the previous perceptron and multiplies it by the weight
    #Then adds the bias and uses the activation function
    #Stores the output in the output variable
    def findOutput(self):
        value = 0

        #The sum of all the weights times the output of the previous perceptron
        for connection in self.backwardsConnections:
            value += connection.getWeight() * connection.previousPerceptron.getOutput()

        #Add the bias
        if not self.isOutputLayer:
            value += self.bias

        #Use the activation function
        self.output = self.activationFunction(value)


    #Updates the bias of the perceptron
    def updateBias(self, delta):
        self.bias += delta

    def getBias(self):
        return self.bias

    #For the input layer it needs to be given the first value
    def giveOutput(self, value):
        self.output = value

    def getWeights(self):
        weights = []
        for connection in self.backwardsConnections:
            weights.append(connection.getWeight())

        return weights

    def __repr(self):
        output = "Bias: " + str(self.bias) \
            + " Output: " + str(self.output) \
            + " Backwards Connections: " \
            + str(self.backwardsConnections) \
            + " Forwards Connections: " \
            + str(self.forwardsConnections) \
            + " Is Output Layer: " + str(self.isOutputLayer)
        return output


    def __str__(self):
        output = "Bias: " + str(self.bias) \
            + " Output: " + str(self.output) \
            + " Backwards Connections: " \
            + str(self.backwardsConnections) \
            + " Forwards Connections: " \
            + str(self.forwardsConnections) \
            + " Is Output Layer: " + str(self.isOutputLayer)
        return output

    def calculateError(self, expected):
        return expected - self.output

class NeuralNetwork:
    def __init__(self, shape, learningRate = 0.1):
        self.layers = []
        self.learningRate = learningRate

        #Build the network
        for i in range(len(shape)):
            layer = []
            isOutputLayer = False

            if i == len(shape) - 1:
                isOutputLayer = True

            for i in range(shape[i]):
                layer.append(Perceptron(isOutputLayer))

            self.layers.append(layer)


        #Connect the layers
        for i in range(1, len(self.layers)):
            for perceptron in self.layers[i]:
                perceptron.makeConnections(self.layers[i-1])


    def feedForward(self, input):
        if len(input) != len(self.layers[0]):
            print("Error: Input size does not match input layer size")
            return

        for i in range(len(self.layers)):
            if i == 0:
                for j in range(len(self.layers[i])):
                    self.layers[i][j].giveOutput(input[j])

            else:
                for perceptron in self.layers[i]:
                    perceptron.findOutput()

        output = []

        for perceptron in self.layers[-1]:
            output.append(perceptron.getOutput())


    def getOutput(self):
        output = []

        for perceptron in self.layers[-1]:
            output.append(perceptron.getOutput())

        return output


    def costFunction(self, expected, actual):
        cost = 0

        for i in range(len(expected)):
            cost += (expected[i] - actual[i])**2

        return cost


    def getWeightsAndBiases(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                print("Perceptron", j, "in layer", i)
                print("Bias =", self.layers[i][j].getBias())
                print("Weights =", self.layers[i][j].getWeights())


    def getPrediction(self, input):
        output = []
        self.feedForward(input)

        for perceptron in self.layers[-1]:
            ans = perceptron.getOutput()
            output.append(ans)

        return output

    def getLayer(self, row):
        return self.layers[row]

    def getOutputs(self):
        for i in range(0, len(self.layers)):
            for j in range(len(self.layers[i])):
                print("Perceptron ", j, "has output ", self.layers[i][j].getOutput())

    def getData(self):
        for i in range(len(self.layers)):
            print("\nLayer", i)
            for j in range(len(self.layers[i])):
                print("Perceptron", j)
                print("Bias =", self.layers[i][j].getBias())
                print("Weights =", self.layers[i][j].getWeights())
                print("Output =", self.layers[i][j].getOutput())


    def backPropagation(self, target):

        output = self.getOutput()
        # print("Num layers =", len(self.layers))

        #Loops through all the layers backwards
        for i in range(len(self.layers) -1, -1, -1):

            #For each perceptron in the layer
            for j in range(len(self.layers[i])):

                #Update the Biases, ignores the output layer
                try:
                    self.layers[i][j].updateBias(self.learningRate * (target[i] - output[i]) * output[i] * (1 - output[i]) * self.layers[i+1][j].getOutput())
                    # print("Updated layer", i, "perceptron", j)
                except:
                    pass

                #Updates the weights of the connections between it and layer after it
                for k in range(len(self.layers[i][j].backwardsConnections)):
                    # print("\nChecking layer", i, "perceptron", j, "backwards connection to ", k)
                    # print("Running a backwards conenction update")
                    # print("Learning rate =", self.learningRate)
                    # print("A=", (target[i] - output[i]))
                    # print("B=", output[i] * (1 - output[i]))
                    # print("C=", self.layers[i-1][j].getOutput())
                    # print("D=", self.layers[i][j].backwardsConnections[k].previousPerceptron.getOutput())
                    # print("Change being made = ", self.learningRate
                    #     * (target[i] - output[i])
                    #     * output[i] * (1 - output[i])
                    #     * self.layers[i-1][j].getOutput()
                    #     * self.layers[i][j].backwardsConnections[k].previousPerceptron.getOutput()
                    # )
                    try:
                        self.layers[i][j].backwardsConnections[k].updateWeight(
                            self.learningRate
                            * (target[i] - output[i])
                            * output[i] * (1 - output[i])
                            * self.layers[i-1][j].getOutput()
                            * self.layers[i][j].backwardsConnections[k].previousPerceptron.getOutput())

                        # print("Updated the perceptron weight to", self.layers[i][j].backwardsConnections[k].getWeight())
                    except:
                        pass

# ----- MAIN -----

#Get the data
data = importData(['archive/mnist_train.csv', 'archive/mnist_test.csv'])
train, val, test = trainValTestSplit(data, 0.8, 0.1)

train_labels, train_data = getLabels(train)
val_labels, val_data = getLabels(val)
test_labels, test_data = getLabels(test)

inputLayerSize = len(train_data[0])
outputLayerSize = 10

zeroHiddenLayers = NeuralNetwork([inputLayerSize, outputLayerSize])
oneHiddenLayer = NeuralNetwork([inputLayerSize, 128, outputLayerSize])
twoHiddenLayers = NeuralNetwork([inputLayerSize, 128, 32, outputLayerSize])
threeHiddenLayers = NeuralNetwork([inputLayerSize, 128, 64, 32, outputLayerSize])

epochs = 50

# # For zero layers
# print("Zero Hidden Layers")
# for i in range(epochs):
#     #Train the network
#     for j in range(len(train_data)):
#         zeroHiddenLayers.feedForward(train_data[j])
#         zeroHiddenLayers.backPropagation(train_labels[j])

#     correct = 0
#     for j in range(len(val_data)):
#         if val_labels[j] == zeroHiddenLayers.getPrediction(val_data[j]):
#             correct += 1

#     print("Epoch", i, "Accuracy =", correct/len(val_data))

# print("One hidden layer")
# # For one layer
# for i in range(epochs):
#     #Train the network
#     for j in range(len(train_data)):
#         oneHiddenLayer.feedForward(train_data[j])
#         oneHiddenLayer.backPropagation(train_labels[j])

#     correct = 0
#     for j in range(len(val_data)):
#         if val_labels[j] == oneHiddenLayer.getPrediction(val_data[j]):
#             correct += 1

#     print("Epoch", i, "Accuracy =", correct/len(val_data))

# print("Two hidden layers")
# # For two layers
# for i in range(epochs):
#     #Train the network
#     for j in range(len(train_data)):
#         twoHiddenLayers.feedForward(train_data[j])
#         twoHiddenLayers.backPropagation(train_labels[j])

#     correct = 0
#     for j in range(len(val_data)):
#         if val_labels[j] == twoHiddenLayers.getPrediction(val_data[j]):
#             correct += 1

#     print("Epoch", i, "Accuracy =", correct/len(val_data))


print("Three hidden layers")
# For three layers
for i in range(epochs):
    #Train the network
    for j in range(len(train_data)):
        threeHiddenLayers.feedForward(train_data[j])
        threeHiddenLayers.backPropagation(train_labels[j])

    correct = 0
    for j in range(len(val_data)):
        if val_labels[j] == threeHiddenLayers.getPrediction(val_data[j]):
            correct += 1

    print("Epoch", i, "Accuracy =", correct/len(val_data))