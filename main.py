import numpy as np
import matplotlib.pyplot as plt
import zeroHiddenLayers
import oneHiddenLayer
import twoHiddenLayers
import threeHiddenLayers


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



print("Collecting Data")
data = load_data()
train_data, train_labels, val_data, val_labels, test_data, test_labels = formatData(data)
print("Data collected")

accuracies = []

#No hidden layers
zeroLayers = zeroHiddenLayers.ZeroHiddenLayerNeuralNetwork(train_data, train_labels, val_data, val_labels)
zeroLayers.train()
accuracies.append( [0, zeroLayers.test(test_data, test_labels)])

#One hidden layer
# print("Training Model with one hidden layer")
# oneLayer = oneHiddenLayer.OneHiddenLayerNeuralNetwork(train_data, train_labels, val_data, val_labels)
# oneLayer.train()
# accuracies.append( [1, oneLayer.test(test_data, test_labels)])

# #Two hidden layers
print("Training Model with two hidden layers")
twoLayers = twoHiddenLayers.TwoHiddenLayerNeuralNetwork(train_data, train_labels, val_data, val_labels)
twoLayers.train()
accuracies.append( [2, twoLayers.test(test_data, test_labels)] )

#Three hidden layers
print("Training Model with three hidden layers")
threeLayers = threeHiddenLayers.ThreeHiddenLayerNeuralNetwork(train_data, train_labels, val_data, val_labels)
threeLayers.train()
accuracies.append( [3, threeLayers.test(test_data, test_labels)])


# Final print out
print("Most Accurate Model")
accuracies.sort(key=lambda x: x[1], reverse=True)
for row in accuracies:
    print("Number of Hidden Layers: " + str(row[0]) + " Accuracy: " + str(row[1]))
