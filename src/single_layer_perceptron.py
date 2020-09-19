from matplotlib import pyplot as plt 
import numpy as np
import random
from datetime import datetime

class SimplePerceptron:
    
    def __init__(self, alpha=0.01, iterations=100):
        self.alpha = alpha
        self.iterations = iterations
    
    def create_plot(self, data, weights, operand):
        fig,ax = plt.subplots()
        ax.set_title(operand)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        map_min = -1.5
        map_max = 2
        res = 0.5
        x = np.linspace(-1.5, 2, 10) # te devuelve un intervalo entre -1.5 y 2 con numeros espaciados iguales para graficar el hiperplano
        plt.plot(x, -((weights[0] + weights[1] * x) / weights[2]), '-g', label='Hiperplano')

        positives = [[],[]]
        negatives = [[],[]]
        for i in range(len(data)):
            x = data[i][1]
            y = data[i][2]
            desired  = data[i][-1]
            if desired == 1:
                positives[0].append(x)
                positives[1].append(y)
            else:
                negatives[0].append(x)
                negatives[1].append(y)

        plt.xlim(map_min,map_max - 0.5)
        plt.ylim(map_min,map_max - 0.5)

        plt.scatter(negatives[0], negatives[1], s = 40.0, c = 'r', label = 'Proyección w < 0')
        plt.scatter(positives[0], positives[1], s = 40.0, c = 'b', label = 'Proyección w > 0')

        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()
        return

    def step(self, x): # función de activación escalón
        if x > 0.0: return 1.0 
        if x < 0.0: return -1.0
        else: return 0.0 

    def get_activation(self, xi, weights):
        excitation = 0.0 # acá voy calculando la excitación
        for i,w in zip(xi, weights):
            excitation += i * w 
        return self.step(excitation)   

    def algorithm(self, operand):
        data = []
        #                             bias   x     y    out 
        if operand == "AND": data = [[1.0,  1.0,  1.0,  1.0],
                                     [1.0, -1.0,  1.0, -1.0],
                                     [1.0,  1.0, -1.0, -1.0],
                                     [1.0, -1.0, -1.0, -1.0]]
        else: data = [[1.0,  1.0,  1.0, -1.0],
                      [1.0, -1.0,  1.0,  1.0],
                      [1.0,  1.0, -1.0,  1.0],
                      [1.0, -1.0, -1.0, -1.0]]

        weights = np.random.rand(len(data[0]) - 1, 1)
        for epoch in range(self.iterations): # COTA del ppt
            for i in range(len(data)): # tamaño del conjunto de entrenamiento
                activation = self.get_activation(data[i][:-1], weights) # dame toda la fila menos el ultimo elemento => x_i => x0, x1, x2, ...
                error = data[i][-1] - activation # y(1,i_x) - activacion del ppt
                fixed_diff = self.alpha * error
                delta_ws = np.dot(fixed_diff, data[i]) # [fd * d[0], fd * d[1], ...]
                weights = [delta + weight for delta, weight in zip(delta_ws, weights)]
        self.create_plot(data, weights, operand)
        return

