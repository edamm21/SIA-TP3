import numpy as np
import random
from datetime import datetime
from file_reader import Reader
from plotter import Plotter

class SimplePerceptron:
    
    def __init__(self, alpha=0.01, iterations=100, linear=True, beta=0.5):
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

    def sigmoid(self, x):
	    return 1 / (1 + np.exp(-2 * self.beta * x))

    def sigmoid_derivative(self, x):
	    return 2 * self.beta * self.sigmoid(x) * (1 - self.sigmoid(x))

    def get_sum(self, xi, weights):
        sumatoria = 0.0
        for i,w in zip(xi, weights):
            sumatoria += i * w
        return sumatoria

    def get_activation(self, sumatoria):
        if self.linear: return self.step(sumatoria)
        else: return self.sigmoid(sumatoria)

    def update_weights(self, error, row, curr_weights):
        delta_ws = [error * elem for elem in row]
        if not self.linear:
            h = 0.0
            for i,w in zip(row, curr_weights):
                h += i * w 
            derivative = self.tanh_derivative(h)
            delta_ws = [derivative * delta_w_i for delta_w_i in delta_ws]
        return [delta + weight for delta, weight in zip(delta_ws, curr_weights)]

    def algorithm(self, operand):
        data = []
        r = None
        #                             bias   x     y    out 
        if operand == 'AND': data = [[1.0,  1.0,  1.0,  1.0],
                                     [1.0, -1.0,  1.0, -1.0],
                                     [1.0,  1.0, -1.0, -1.0],
                                     [1.0, -1.0, -1.0, -1.0]]
        elif operand == 'XOR': data = [ [1.0,  1.0,  1.0, -1.0],
                                        [1.0, -1.0,  1.0,  1.0],
                                        [1.0,  1.0, -1.0,  1.0],
                                        [1.0, -1.0, -1.0, -1.0]]
        elif operand == 'TEST': 
            r = Reader('TEST')
            data = r.readFile() # agarramos los datos de los txt
        else:
            r = Reader('Ej2')
            data = r.readFile() # agarramos los datos de los txt

        weights = np.random.rand(len(data[0]) - 1, 1)
        error_min = 20
        total_error = 1
        w_min = None
        error_per_epoch = []
        for epoch in range(self.iterations): # COTA del ppt
            if total_error > 0:
                total_error = 0
                #if (epoch % 100 == 0):
                #    weights = np.random.rand(len(data[0]) - 1, 1)
                for i in range(len(data)): # tamaño del conjunto de entrenamiento
                    sumatoria = self.get_sum(data[i][:-1], weights) # dame toda la fila menos el ultimo elemento => x_i => x0, x1, x2, ...
                    activation = self.get_activation(sumatoria) 
                    error = data[i][-1] - activation # y(1,i_x) - activacion del ppt
                    fixed_diff = self.alpha * error
                    derivative = 1.0
                    if not self.linear:
                        derivative = self.sigmoid_derivative(sumatoria)
                    const = fixed_diff * derivative
                    for j in range(len(weights)):
                        weights[j] = weights[j] + (const * data[i][j])
                    total_error += abs(error)
                error_per_epoch.append(total_error/len(data))
                if total_error < error_min:
                    error_min = total_error
                    w_min = weights
            else:
                break
        plotter = Plotter()
        if operand != 'Ej2':
            if len(w_min) != 0 or w_min != None : plotter.create_plot_ej1(data, w_min, operand)
            else: plotter.create_plot_ej1(data, weights, operand)
        else:
            r = Reader('Ej2') # por las dudas
            test_data = r.readFile(test=True)
            self.test_perceptron(test_data, w_min)
            print('Non linear data post analysis:')
            print('epochs: {}'.format(epoch + 1))
            plotter.create_plot_ej2(error_per_epoch)
        return

    def test_perceptron(self, test_data, weights):
        print('Testing perceptron...')
        element_count = 0
        print('+-------------------+-------------------+')
        print('|   Desired output  |   Perceptron out  |')
        print('+-------------------+-------------------+')
        for row in test_data:
            sumatoria = self.get_sum(row[:-1], weights) # dame toda la fila menos el ultimo elemento => x_i => x0, x1, x2, ...
            perceptron_output = self.get_activation(sumatoria) 
            element_count += 1
            print('|{}|{}|'.format(row[-1], perceptron_output[0]))
        print('Analysis finished')

