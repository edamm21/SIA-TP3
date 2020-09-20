import numpy as np
import random
from datetime import datetime
from file_reader import Reader
from plotter import Plotter

class SimplePerceptron:
    
    def __init__(self, alpha=0.01, iterations=100, linear=True, beta=0.1):
        self.alpha = alpha
        self.iterations = iterations
        self.linear = linear
        self.beta = beta

    def step(self, x): # funcion de activacion escalon
        if x > 0.0: return 1.0 
        if x < 0.0: return -1.0
        else: return 0.0 

    def tanh(self, x):
        return np.tanh(self.beta * x)

    def tanh_derivative(self, x):
        return self.beta * (1.0 - self.tanh(x) ** 2)

    def get_activation(self, xi, weights):
        excitation = 0.0 # aca voy calculando la excitacion
        for i,w in zip(xi, weights):
            excitation += i * w 
        if self.linear: return self.step(excitation)
        else: return self.tanh(excitation)

    def update_weights(self, error, row, curr_weights):
        print(error)
        print(row)
        delta_ws = [error * elem for elem in row]
        if not self.linear:        
            delta_ws = [self.tanh_derivative(delta_w_i) for delta_w_i in delta_ws]
        return [delta + weight for delta, weight in zip(delta_ws, curr_weights)]

    def algorithm(self, operand):
        data = []
        #                             bias   x     y    out 
        if operand == 'AND': data = [[1.0,  1.0,  1.0,  1.0],
                                     [1.0, -1.0,  1.0, -1.0],
                                     [1.0,  1.0, -1.0, -1.0],
                                     [1.0, -1.0, -1.0, -1.0]]
        elif operand == 'XOR': data = [[1.0,  1.0,  1.0, -1.0],
                                          [1.0, -1.0,  1.0,  1.0],
                                          [1.0,  1.0, -1.0,  1.0],
                                          [1.0, -1.0, -1.0, -1.0]]
        else:
            r = Reader('Ej2')
            data = r.readFile() # agarramos los datos de los txt

        weights = np.random.rand(len(data[0]) - 1, 1)
        error_min = 20
        total_error = 1
        for epoch in range(self.iterations): # COTA del ppt
            if total_error > 0:
                total_error = 0
                if (epoch % 100 == 0):
                    weights = np.random.rand(len(data[0]) - 1, 1)
                for i in range(len(data)): # tamaño del conjunto de entrenamiento
                    activation = self.get_activation(data[i][:-1], weights) # dame toda la fila menos el ultimo elemento => x_i => x0, x1, x2, ...
                    error = data[i][-1] - activation # y(1,i_x) - activacion del ppt
                    fixed_diff = self.alpha * error
                    weights = self.update_weights(fixed_diff, data[i], weights)
                    total_error += abs(error)
                if total_error < error_min:
                    error_min = total_error
                    w_min = weights
                else:
                    break
        plotter = Plotter()
        plotter.create_plot_ej1(data, weights, operand)
        return

