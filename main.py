#!/usr/bin/env python3
import scipy.special
import numpy as np


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
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

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hinput = np.dot(self.wih, inputs)
        houtput = self.activation_function(hinput)

        oinput = np.dot(self.who, houtput)
        ooutput = self.activation_function(oinput)

        return ooutput
