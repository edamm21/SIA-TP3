from matplotlib import pyplot as plt 
import numpy as np
import random
from datetime import datetime

class MultiLayerPerceptron:

    def __init__(self, alpha=0.01, iterations=100, hidden_layers=1):
        self.alpha = alpha
        self.iterations = iterations
        self.hidden_layers = hidden_layers
        self.total_layers = hidden_layers + 2 # input + hidden + output
        self.layer_real_results = [None] * self.total_layers
        self.layer_activations = [None] * self.total_layers
        self.layer_weights = [[None] * self.total_layers]
        self.deltas_per_layer = [None] * self.total_layers

    def algorithm(self, problem):
        
        #                           bias    x     y    out
        if problem == "XOR": data = [[1.0,  1.0,  1.0, -1.0],
                                     [1.0, -1.0,  1.0,  1.0],
                                     [1.0,  1.0, -1.0,  1.0],
                                     [1.0, -1.0, -1.0, -1.0]]
        self.layer_weights = [np.random.rand(len(data[0]) - 1, 1)] * (self.total_layers)
        for epoch in range(self.iterations):
            error = 0
            for row in data:
                expected_result_in_row = row[-1]
                self.advance_in_network(row)
                self.backpropagate(row)
                error += (expected_result_in_row - self.layer_activations[self.total_layers - 1]) ** 2
            self.error = error
        
    def advance_in_network(self, row):
        for element in range(len(row)):
            self.layer_real_results[element] = np.dot(row, self.layer_weights[element])
            self.layer_activations[element] = self.sigmoid(self.layer_real_results[element])
        return self.layer_activations[self.total_layers - 1]
    
    def backpropagate(self, row):
        error = (row[-1] - self.layer_activations[self.total_layers - 1])
        for layer in range(self.total_layers - 1, -1, -1):
            current_real_result = self.layer_real_results[layer]
            if layer == self.total_layers - 1:
                self.deltas_per_layer[layer] = error * self.derivative_sigmoid(self.layer_real_results[layer])
            else:
                dot_product = np.dot(self.deltas_per_layer[layer + 1], self.layer_weights[layer + 1].T)
                self.deltas_per_layer[layer] = self.derivative_sigmoid(current_real_result) * dot_product[:, :-1]
        for layer in range(self.total_layers):
            if layer == 0: activation = row
            else: activation = self.layer_activations[layer - 1]
            self.deltas_per_layer[layer] = self.alpha * np.dot(((activation).T), self.deltas_per_layer[layer])
        for layer in range(self.total_layers - 1, -1, -1):
            self.layer_weights[layer] = self.layer_weights[layer] + self.deltas_per_layer[layer]
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
     
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))
    

            

