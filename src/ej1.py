import numpy as np
import random
from datetime import datetime
from file_reader import Reader
from plotter import Plotter

class SimplePerceptron:
    
    def __init__(self, alpha=0.01, iterations=100, adaptive=False):
        self.alpha = alpha
        self.initial_alpha = alpha
        self.iterations = iterations
        self.adaptive = adaptive

    def adjust_learning_rate(self, errors_so_far):
        if(len(errors_so_far) > 10):
            last_10_errors = errors_so_far[-10:]
            print(last_10_errors)
            booleans = []
            for i in range(len(last_10_errors) - 1):
                booleans.append(last_10_errors[i] > last_10_errors[i + 1])
            print(booleans)
            if all(booleans):
                self.alpha += 0.001
            elif not all(booleans):
                self.alpha -= 0.01 * self.alpha

    def error_function(self, sqr_errors_sum):
        if isinstance(sqr_errors_sum, list):
            return (0.5 * (sqr_errors_sum))[0] 
        else:
            return 0.5 * (sqr_errors_sum)

    def step(self, x): # funcion de activacion escalon
        if x > 0.0: return 1.0 
        if x < 0.0: return -1.0
        else: return 0.0 

    def get_sum(self, xi, weights):
        sumatoria = 0.0
        for i,w in zip(xi, weights):
            sumatoria += i * w
        return sumatoria

    def get_activation(self, sumatoria):
        return self.step(sumatoria)

    def algorithm(self, operand):
        data = []
        #                             bias   x     y    out 
        if operand == 'AND': data = [[1.0,  1.0,  1.0,  1.0],
                                     [1.0, -1.0,  1.0, -1.0],
                                     [1.0,  1.0, -1.0, -1.0],
                                     [1.0, -1.0, -1.0, -1.0]]
        elif operand == 'XOR': data = [ [1.0,  1.0,  1.0, -1.0],
                                        [1.0, -1.0,  1.0,  1.0],
                                        [1.0,  1.0, -1.0,  1.0],
                                        [1.0, -1.0, -1.0, -1.0]]
        weights = np.random.rand(len(data[0]) - 1, 1)
        error_min = len(data) * 2
        error_this_epoch = 1
        w_min = weights
        error_per_epoch = []
        plotter = Plotter()
        for epoch in range(self.iterations): # COTA del ppt
            if error_this_epoch > 0:
                total_error = 0
                #if (epoch % 100 == 0):
                #    weights = np.random.rand(len(data[0]) - 1, 1)
                for i in range(len(data)): # tamano del conjunto de entrenamiento
                    sumatoria = self.get_sum(data[i][:-1], weights) # dame toda la fila menos el ultimo elemento => x_i => x0, x1, x2, ...
                    activation = self.get_activation(sumatoria) 
                    error = data[i][-1] - activation # y(1,i_x) - activacion del ppt
                    fixed_diff = self.alpha * error 
                    for j in range(len(weights)):
                        weights[j] = weights[j] + (fixed_diff * data[i][j])
                    total_error += error**2
                error_this_epoch = self.error_function(total_error)    
                error_per_epoch.append(error_this_epoch)
                if self.adaptive and epoch % 10 == 0:
                    self.adjust_learning_rate(error_per_epoch)
                if error_this_epoch < error_min:
                    error_min = error_this_epoch
                    w_min = weights
            else:
                break
        print('Linear data post analysis:')
        print('epochs: {}'.format(epoch + 1))
        print('Initial alpha: {}'.format(self.initial_alpha))
        print('End alpha: {}'.format(self.alpha))
        print('Best ever error: {}'.format(error_min))
        if len(w_min) != 0 or w_min != None : plotter.create_plot_ej1(data, w_min, operand)
        else: plotter.create_plot_ej1(data, weights, operand)
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
            print('|{}|{}|'.format(row[-1], perceptron_output))
        print('Analysis finished')

