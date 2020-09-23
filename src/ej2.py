import numpy as np
import random
from datetime import datetime
from file_reader import Reader
from plotter import Plotter

class SimplePerceptronEj2:
    
    def __init__(self, alpha=0.01, iterations=100, beta=0.5, adaptive=False):
        self.alpha_linear = alpha
        self.alpha_non_linear = alpha
        self.initial_alpha_linear = alpha
        self.initial_alpha_non_linear = alpha
        self.iterations = iterations
        self.beta = beta
        self.adaptive = adaptive

    def adjust_learning_rate(self, errors_so_far, linear):
        if(len(errors_so_far) > 10):
            last_10_errors = errors_so_far[-10:]
            booleans = []
            for i in range(len(last_10_errors) - 1):
                booleans.append(last_10_errors[i] > last_10_errors[i + 1])
            if all(booleans) and linear:
                self.alpha_linear += 0.001
            elif all(booleans) and not linear:
                self.alpha_non_linear += 0.001
            elif not all(booleans) and linear:
                self.alpha_linear -= 0.01 * self.alpha_linear
            elif not all(booleans) and not linear:
                self.alpha_non_linear -= 0.01 * self.alpha_non_linear

    def error_function(self, sqr_errors_sum):
        if isinstance(sqr_errors_sum, list):
            return (0.5 * (sqr_errors_sum))[0] 
        else:
            return 0.5 * (sqr_errors_sum)

    def identity(self, x): # funcion de activacion escalon
        return x

    def sigmoid(self, x):
	    return 1 / (1 + np.exp(-2 * self.beta * x))

    def sigmoid_derivative(self, x):
	    return 2 * self.beta * self.sigmoid(x) * (1 - self.sigmoid(x))

    def get_sum(self, xi, weights):
        sumatoria = 0.0
        for i,w in zip(xi, weights):
            sumatoria += i * w
        return sumatoria

    def get_activation(self, sumatoria, linear):
        if linear: return self.identity(sumatoria)[0]
        else: return self.sigmoid(sumatoria)[0]

    def algorithm(self, operand):
        r = Reader('Ej2')
        data_linear, data_non_linear, test_data_linear, test_data_non_linear, max_, min_ = r.readFile() # agarramos los datos de los txt
        #test_data_linear, test_data_non_linear, max_out, min_out = r.readFile(test=True)
        initial_weights = np.random.rand(len(data_linear[0]) - 1, 1)
        weights_linear     = initial_weights.copy()
        weights_non_linear = initial_weights.copy()
        error_min_linear     = len(data_linear) * 2
        error_min_non_linear = len(data_non_linear) * 2
        error_this_epoch_linear     = 1
        error_this_epoch_non_linear = 1
        w_min_linear     = initial_weights.copy()
        w_min_non_linear = initial_weights.copy()
        error_per_epoch_linear     = []
        error_per_epoch_non_linear = []
        alpha_per_epoch_linear     = []
        alpha_per_epoch_non_linear = []
        plotter = Plotter()
        test_error_per_epoch_linear     = []
        test_error_per_epoch_non_linear = []
        for epoch in range(self.iterations): # COTA del ppt
            if error_this_epoch_linear > 0 and error_min_non_linear > 0:
                total_error_linear = total_error_non_linear = 0
                for i in range(len(data_linear)):
                    sumatoria_linear = self.get_sum(data_linear[i][:-1], weights_linear)
                    sumatoria_non_linear = self.get_sum(data_non_linear[i][:-1], weights_non_linear) 
                    activation_linear = self.get_activation(sumatoria_linear, linear=True)
                    activation_non_linear = self.get_activation(sumatoria_non_linear, linear=False)
                    error_linear = data_linear[i][-1] - activation_linear
                    error_non_linear = data_non_linear[i][-1] - activation_non_linear
                    fixed_diff_linear = self.alpha_linear * error_linear
                    fixed_diff_non_linear = self.alpha_non_linear * error_non_linear                
                    derivative_linear = 1.0
                    derivative_non_linear = self.sigmoid_derivative(sumatoria_non_linear)
                    const_linear = fixed_diff_linear * derivative_linear
                    const_non_linear = fixed_diff_non_linear * derivative_non_linear[0]
                    for j in range(len(weights_linear)):
                        weights_linear[j] = weights_linear[j] + (const_linear * data_linear[i][j])
                    for j in range(len(weights_non_linear)):
                        weights_non_linear[j] = weights_non_linear[j] + (const_non_linear * data_non_linear[i][j])
                    total_error_linear += error_linear**2
                    total_error_non_linear += self.denormalize(error_non_linear, max_, min_)**2
                error_this_epoch_linear = self.error_function(total_error_linear)/len(data_linear)    
                error_per_epoch_linear.append(error_this_epoch_linear)
                error_this_epoch_non_linear = self.error_function(total_error_non_linear)/len(data_non_linear)    
                error_per_epoch_non_linear.append(error_this_epoch_non_linear)
                if self.adaptive and epoch % 10 == 0:
                    self.adjust_learning_rate(error_per_epoch_linear, linear=True)
                    self.adjust_learning_rate(error_per_epoch_non_linear, linear=False)
                alpha_per_epoch_linear.append(self.alpha_linear)
                alpha_per_epoch_non_linear.append(self.alpha_non_linear)
                if epoch == 0:
                    error_min_linear = error_this_epoch_linear
                if error_this_epoch_linear < error_min_linear:
                    error_min_linear = error_this_epoch_linear
                    w_min_linear = weights_linear
                if error_this_epoch_non_linear < error_min_non_linear:
                    error_min_non_linear = error_this_epoch_non_linear
                    w_min_non_linear = weights_non_linear
                test_error_per_epoch_linear.append(self.test_perceptron(test_data_linear, w_min_linear, linear=True, max_out=None, min_out=None, print_=False))
                test_error_per_epoch_non_linear.append(self.test_perceptron(test_data_non_linear, w_min_non_linear, linear=False, max_out=max_, min_out=min_, print_=False))
            else:
                break
        
        print('*************** RESULTS ***************')
        print('Analysis for training set:')
        print('Epochs: {}'.format(epoch + 1))
        print('Initial alpha linear: {}'.format(self.initial_alpha_linear))
        print('Initial alpha non linear: {}'.format(self.initial_alpha_non_linear))
        print('End alpha linear: {}'.format(self.alpha_linear))
        print('End alpha non linear: {}'.format(self.alpha_non_linear))
        print('***************************************')
        self.test_perceptron(test_data_linear, w_min_linear, linear=True, max_out=None, min_out=None, print_=True)
        print('***************************************')
        self.test_perceptron(test_data_non_linear, w_min_non_linear, linear=False, max_out=max_, min_out=min_, print_=True)
        print('***************************************')
        plotter.create_plot_ej2(error_per_epoch_linear, test_error_per_epoch_linear, alpha_per_epoch_linear, linear=True)
        plotter.create_plot_ej2(error_per_epoch_non_linear, test_error_per_epoch_non_linear, alpha_per_epoch_non_linear, linear=False)
        return

    def test_perceptron(self, test_data, weights, linear, max_out, min_out, print_):
        if print_: print('Testing perceptron {}...'.format(('linear' if linear else 'non linear')))
        error = 0.0
        error_accum = 0.0
        if print_:
            print('+-------------------+-------------------+')
            print('|   Desired output  |   Perceptron out  |')
            print('+-------------------+-------------------+')
        for row in test_data:
            sumatoria = self.get_sum(row[:-1], weights)
            perceptron_output = self.get_activation(sumatoria, linear=linear)
            if not linear:
                denorm_real = self.denormalize(row[-1], max_out, min_out)
                denorm_perc = self.denormalize(perceptron_output, max_out, min_out)
                if print_: print('|{:19f}|{:19f}|'.format(denorm_real, denorm_perc))
                diff = denorm_real - denorm_perc
                error_accum += abs(diff)
                error += (diff)**2
            else:
                if print_: print('|{:19f}|{:19f}|'.format(row[-1], perceptron_output))
                diff = row[-1] - perceptron_output
                error_accum += abs(diff)
                error += (diff)**2
        if print_: 
            print('+-------------------+-------------------+')
            print('Test finished')
            print('Error avg {}: {}'.format('linear' if linear else 'non linear', error_accum/len(test_data)))
        return error/len(test_data)

    def denormalize(self, x, max_, min_):
        return x * (max_ - min_) + min_
