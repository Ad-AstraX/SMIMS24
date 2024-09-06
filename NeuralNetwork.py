#!/usr/bin/env python3
import ast

import scipy.special
import numpy as np
import json
import sys

from twisted.web.html import output


class NeuralNetwork:
    def __init__(self, inodes, hnodes, onodes, lr, initialweights=None):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        #self.wih = np.array(initialweights["wih"])
        #self.who = np.array(initialweights["who"])
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = lr
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, reward,action):
        inputs = np.array(inputs_list, ndmin=2).T
        #targets = np.array(targets_list, ndmin=2).T

        hinput = np.dot(self.wih, inputs)
        houtput = self.activation_function(hinput)

        oinput = np.dot(self.who, houtput)
        ooutput = self.activation_function(oinput)

        eho = ooutput*0.1*reward
        print(action)
        if action!=None:
            eho*=np.argmax(action)
        eih = np.dot(self.who.T, eho)

        self.who += np.dot(self.lr * eho * ooutput * (1 - ooutput), houtput.T)
        self.wih += np.dot(self.lr * eih * houtput * (1 - houtput), inputs.T)
    def train_short(self, inputs_list, reward,state_new):
        #inputs = np.array(inputs_list, ndmin=2).T

        hinput = np.dot(self.wih, inputs_list)
        houtput = self.activation_function(hinput)

        oinput = np.dot(self.who, houtput)
        ooutput = self.activation_function(oinput)
        #inputs1 = np.array(state_new, ndmin=2).T

        hinput1 = np.dot(self.wih, state_new)
        houtput1 = self.activation_function(hinput1)

        oinput1 = np.dot(self.who, houtput1)
        ooutput1 = self.activation_function(oinput1)

        eho = (oinput1-oinput)*reward
        print(eho)
        eih = np.dot(self.who.T, eho)
        print(houtput1)

        self.who += self.lr * np.dot(eho * ooutput1 * (1 - ooutput1), ooutput1.T)
        self.wih += self.lr * np.dot(eih * houtput1 * (1 - houtput1), houtput1.T)

    def train_multiple_epochs(self, input, target, epochs):
        progress = 0
        for e in range(epochs):
            for i in range (len(input)):
                self.train(input[i], target[i])

                progress += 1
                sys.stdout.write ("\rProgressbar: " + str(int(progress/(epochs*len(input))*100)) + "%   ")
                for i in range(int(progress/(epochs*len(input))*100)):
                    sys.stdout.write ("=")

                sys.stdout.flush()


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

        neuralNetwork = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, 0.005, weight)
    except FileNotFoundError:
        pass

    try:
        with open("weights.json", 'x') as file:
            wih = np.random.normal(0.0, pow(inputNodes, -0.5), (hiddenNodes, inputNodes))
            who = np.random.normal(0.0, pow(hiddenNodes, -0.5), (outputNodes, hiddenNodes))

            weights = {"wih" : wih.tolist(), "who" : who.tolist()}
            json.dump(weights, file, indent=6)

        neuralNetwork = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, 0.005, weights)

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
        print("success rate: ", rate/len(input))

#create_NN()