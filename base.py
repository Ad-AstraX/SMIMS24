#!/usr/bin/env python3
import scipy.special
import numpy as np

class NeuralNetwork:
    def __init__(self, inodes, hnodes, onodes, lr, weights_path):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        self.weights_path = weights_path
        if (self.weights_path / 'wih.npy').exists():
            self.wih = np.load(weights_path / 'wih.npy')
            self.who = np.load(weights_path / 'who.npy')
        else:
            self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

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

    def train_multiple_epochs(self, inputs, target, epochs):
        for e in range(epochs):
            for i in range (1,200):#len(inputs)):
                print(self.query(inputs[i]))
                self.train(inputs[i], target[i])
                print(i,self.query(inputs[i]))

        np.save(self.weights_path / 'wih.npy', self.wih)
        np.save(self.weights_path / 'who.npy', self.who)


    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hinput = np.dot(self.wih, inputs)
        houtput = self.activation_function(hinput)

        oinput = np.dot(self.who, houtput)
        ooutput = self.activation_function(oinput)

        return ooutput
