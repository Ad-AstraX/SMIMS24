#!/usr/bin/env python3
import ast

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
            for i in range (len(input)):
                self.train(input[i], target[i])
                if (e == 0 or e == 99):
                    output = self.query(input[i]).T[0].tolist()
                    print("Neural Network prediction", output.index(max(output)), "target:", target[i])

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
    inputNodes = 784
    hiddenNodes = 100
    outputNodes = 10

    try:
        with open('weights.json', 'r') as file:
            data = json.load(file)

        weight = ast.literal_eval(json.dumps(data, indent=4))

        neuralNetwork = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, 2, weight)
    except FileNotFoundError:
        pass

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
        test_file = open("material/mnist_test_10.csv", 'r')
        test_list = test_file.readlines()
        test_file.close()
        input = []
        target = []

        for number in test_list:
            all_values = number.split(',')
            input += [(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01]
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            target += [targets]

        rate = 0
        for i in range(0, len(input)):
            output = neuralNetwork.query(input[i]).T[0].tolist()
            target_out = target[i].T.tolist()
            if (output.index(max(output)) == target_out.index(max(target_out))):
                rate += 1
            print("Neural Network prediction", output.index(max(output)), "target:", target_out.index(max(target_out)))
        print(rate/len(input))



create_NN()