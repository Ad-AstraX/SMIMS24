#!/usr/bin/env python3
import numpy
import scipy.special
import numpy as np
import json


class NeuralNetwork:
    def __init__(self, inodes, hnodes, onodes, lr, initialweights):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        self.wih = np.array(initialweights["wih"])
        self.who = np.array(initialweights["who"])
        #self.wih.split(" ")
        #self.wih.astype(int)

        #print(np.dtype(self.wih))

        self.lr = lr
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hinput = np.dot(self.wih, inputs)
        houtput = self.activation_function(hinput)

        oinput = np.dot(self.who, houtput)
        ooutput = self.activation_function(oinput)

        eho = targets - ooutput
        eih = np.dot(self.who.T, eho)

        self.who += self.lr * np.dot(eho * ooutput * (1 - ooutput), houtput.T)
        self.wih += self.lr * np.dot(eih * houtput * (1 - houtput), inputs.T)

    def train_multiple_epochs(self, input, target, epochs):
        for e in range(epochs):
            self.train(input[e], target[e])

        with open("weights.json", 'w') as file:
            weights = {"wih" : self.wih.tolist(), "who" : self.who.tolist()}
            json.dump(weights, file, indent=6)


    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hinput = np.dot(self.wih, inputs)
        houtput = self.activation_function(hinput)

        oinput = np.dot(self.who, houtput)
        ooutput = self.activation_function(oinput)

        return ooutput

def create_NN():
    inputNodes = 3
    hiddenNodes = 3
    outputNodes = 3

    try:
        with open("weights.json", 'x') as file:
            wih = np.random.normal(0.0, pow(inputNodes, -0.5), (hiddenNodes, inputNodes))
            who = np.random.normal(0.0, pow(hiddenNodes, -0.5), (outputNodes, hiddenNodes))

            weights = {"wih" : wih.tolist(), "who" : who.tolist()}
            json.dump(weights, file, indent=6)

        neuralNetwork = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, 2, weights)

        data_file = open("material/mnist_train_100.csv", 'r')
        data_list = data_file.readlines()
        data_file.close()
        #neuralNetwork.train_multiple_epochs(np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), 1)
        input = []
        target = []

        for number in data_list:
            all_values = number.split(',')
            input += [(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01]
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            target += [targets]

        neuralNetwork.train_multiple_epochs(input, target, 100)

    except FileExistsError:
        #neuralNetwork.train();
        print (5)


create_NN()