import numpy as np
import matplotlib.pyplot as plt

train_path = 'mnist_train.csv'
test_path = 'mnist_test.csv'


def trainValTestSplit(data):
    train_data = data[:int(data.shape[0]*0.6)]
    val_data = data[int(data.shape[0]*0.6):int(data.shape[0]*0.8)]
    test_data = data[int(data.shape[0]*0.8):]

    return train_data, val_data, test_data


def createLabels(y):
    labels = np.zeros((y.shape[0], 10))

    for i in range(y.shape[0]):
        labels[i][int(y[i][0])] = 1

    return labels


def normalize(x):
    x = x / 255
    return x


def load_data():

    #Note that the first row from both files has been deleted
    data_part_one = np.genfromtxt(train_path, delimiter=',')
    data_part_two = np.genfromtxt(test_path, delimiter=',')

    train_data = np.concatenate((data_part_one, data_part_two))
    return train_data


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


class NeuralNetwork:
    def __init__(self, train_data, train_labels, val_data, val_labels, batch = 64, lr = 0.01,  epochs = 50):
        self.input = train_data
        self.target = train_labels
        self.val_input = val_data
        self.val_target = val_labels
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.loss = []
        self.acc = []

        self.init_weights(train_labels.shape[1])


    def init_weights(self, shape):
        self.W1 = np.random.randn(self.input.shape[1],256)
        self.W2 = np.random.randn(self.W1.shape[1],128)
        self.W3 = np.random.randn(self.W2.shape[1], shape)

        self.b1 = np.random.randn(self.W1.shape[1],)
        self.b2 = np.random.randn(self.W2.shape[1],)
        self.b3 = np.random.randn(self.W3.shape[1],)


    def ReLU(self, x):
        return np.maximum(0,x)


    def dReLU(self,x):
        return 1 * (x > 0)


    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)


    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]


    def feedforward(self, value, label):
        assert value.shape[1] == self.W1.shape[0]
        self.z1 = value.dot(self.W1) + self.b1
        self.a1 = self.ReLU(self.z1)

        assert self.a1.shape[1] == self.W2.shape[0]
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.ReLU(self.z2)

        assert self.a2.shape[1] == self.W3.shape[0]
        self.z3 = self.a2.dot(self.W3) + self.b3
        self.a3 = self.softmax(self.z3)

        self.error = self.a3 - label

        return self.a3


    def backprop(self, input):
        dcost = (1/self.batch)*self.error

        DW3 = np.dot(dcost.T,self.a2).T
        DW2 = np.dot((np.dot((dcost),self.W3.T) * self.dReLU(self.z2)).T,self.a1).T
        DW1 = np.dot((np.dot(np.dot((dcost),self.W3.T)*self.dReLU(self.z2),self.W2.T)*self.dReLU(self.z1)).T,input).T

        db3 = np.sum(dcost,axis = 0)
        db2 = np.sum(np.dot((dcost),self.W3.T) * self.dReLU(self.z2),axis = 0)
        db1 = np.sum((np.dot(np.dot((dcost),self.W3.T)*self.dReLU(self.z2),self.W2.T)*self.dReLU(self.z1)),axis = 0)

        assert DW3.shape == self.W3.shape
        assert DW2.shape == self.W2.shape
        assert DW1.shape == self.W1.shape

        assert db3.shape == self.b3.shape
        assert db2.shape == self.b2.shape
        assert db1.shape == self.b1.shape

        self.W3 = self.W3 - self.lr * DW3
        self.W2 = self.W2 - self.lr * DW2
        self.W1 = self.W1 - self.lr * DW1

        self.b3 = self.b3 - self.lr * db3
        self.b2 = self.b2 - self.lr * db2
        self.b1 = self.b1 - self.lr * db1

    def train(self):
        for epoch in range(self.epochs):
            loss = 0
            accuracy = 0
            self.shuffle()

            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.feedforward(self.input[start:end], self.target[start:end])
                self.backprop(self.input[start:end])
                loss += np.mean(self.error**2)

            self.loss.append( loss / (self.input.shape[0] // self.batch))

            for batch in range(self.val_input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.feedforward(self.val_input[start:end], self.val_target[start:end])
                accuracy += np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.val_target[start:end],axis=1)) / self.batch

            self.acc.append( accuracy*100 / (self.val_input.shape[0] // self.batch))
            print("Epoch {} Loss: {} Accuracy: {}%".format(epoch+1,self.loss[-1],self.acc[-1]))


    def plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    def acc_plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.acc)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

    def test(self, data, labels):
        accuracy = 0
        for batch in range(data.shape[0]//self.batch-1):
            start = batch*self.batch
            end = (batch+1)*self.batch
            self.feedforward(data[start:end], labels[start:end])
            accuracy += np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(labels[start:end],axis=1)) / self.batch

        accuracy = accuracy * 100 / (data.shape[0] // self.batch)
        print("Final Accuracy = ", accuracy)


data = load_data()
train_data, train_labels, val_data, val_labels, test_data, test_labels = formatData(data)


NN = NeuralNetwork(train_data, train_labels, val_data, val_labels)
NN.train()
# NN.plot()
NN.test(test_data, test_labels)