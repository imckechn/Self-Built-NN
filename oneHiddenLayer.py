import numpy as np
import matplotlib.pyplot as plt

train_path = 'mnist_train.csv'
test_path = 'mnist_test.csv'


#Splitting the data into train, validation, and test
def trainValTestSplit(data):
    train_data = data[:int(data.shape[0]*0.6)]
    val_data = data[int(data.shape[0]*0.6):int(data.shape[0]*0.8)]
    test_data = data[int(data.shape[0]*0.8):]

    return train_data, val_data, test_data


#Generating labels for the data so it's in the form [0,0,0,1,0...]
def createLabels(originalLabels):
    labels = np.zeros((originalLabels.shape[0], 10))

    for i in range(originalLabels.shape[0]):
        labels[i][int(originalLabels[i][0])] = 1

    return labels


#Helper function to normalize the data so it's all floats between 0 and 1
def normalize(x):
    x = x / 255
    return x


#Grabs the data from the files and combines them into one mega data object
def load_data():

    #Note that the first row from both files has been deleted
    data_part_one = np.genfromtxt(train_path, delimiter=',')
    data_part_two = np.genfromtxt(test_path, delimiter=',')

    train_data = np.concatenate((data_part_one, data_part_two))
    return train_data


#Splits the data into train, validation, and test, then normalizes it, and finally creates labels for it, returns 6 numpy arrays
def formatData(data):
    #Splitting the data into train, validation and test
    train_data, val_data, test_data = trainValTestSplit(data)

    #Normalize train data
    train_normalized = normalize(train_data[:,1:])
    train_labels = createLabels(train_data[:,:1])

    #Normalize validation data
    val_normalized = normalize(val_data[:,1:])
    val_labels = createLabels(val_data[:,:1])

    #Normalize test data
    test_normalized = normalize(test_data[:,1:])
    test_labels = createLabels(test_data[:,:1])

    return train_normalized, train_labels, val_normalized, val_labels, test_normalized, test_labels



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

    # Using relu
    def relu(self, x):
        return np.maximum(0, x)

    # Derivative of relu, returns a copy of the OG array but with 1 or 0 depending on if the value is > 0
    def relu_derivative(self, arr):
        return arr > 0

    # The softmax function for the output layer
    def softmax(self, z):
        z = z - np.max(z, axis=1).reshape(z.shape[0], 1)
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)

    # Push a value through the NN
    def feedforward(self, value, label):
        self.initial_value1 = value.dot(self.weight1) + self.b1
        self.answer1 = self.relu(self.initial_value1)
        self.initial_value2 = self.answer1.dot(self.weight2) + self.b2
        self.answer2 = self.softmax(self.initial_value2)
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
        hidden_error = np.dot(cost, self.weight2.T) * self.relu_derivative(self.initial_value1)
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

data = load_data()
train_data, train_labels, val_data, val_labels, test_data, test_labels = formatData(data)


twoLayers = OneHiddenLayerNeuralNetwork(train_data, train_labels, val_data, val_labels)
twoLayers.train()
# twoLayers.test(test_data, test_labels)

