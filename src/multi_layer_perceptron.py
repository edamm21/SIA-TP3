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

    def create_plot(self, data, weights, operand):
        fig,ax = plt.subplots()
        ax.set_title(operand)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        map_min = -1.5
        map_max = 2
        res = 0.5
        x = np.linspace(-1.5, 2, 10) # te devuelve un intervalo entre -1.5 y 2 con numeros espaciados iguales para graficar el hiperplano
        for w in range(1,3):
            plt.plot(x, -((weights[w][0] + weights[w][1] * x) / weights[w][2]), '-g', label='Hiperplano')

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

        plt.scatter(negatives[0], negatives[1], s = 40.0, c = 'r', label = 'Proyeccion w < 0')
        plt.scatter(positives[0], positives[1], s = 40.0, c = 'b', label = 'Proyeccion w > 0')

        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()
        return

    def g(self, x):
        return 1.0 / (1.0 + np.exp(-x))
 
    def g_derivative(self, x):
        return self.g(x) * (1.0 - self.g(x))

    def h(self, m, i, amount_of_nodes, W, V):
        hmi = 0
        for j in range(0, amount_of_nodes):
            hmi += W[m,i,j] * V[m-1][j]
        return hmi

    def algorithm(self, problem):
        #                           bias    x     y    out
        if problem == "XOR": data = [[1.0,  1.0,  1.0, -1.0],
                                     [1.0, -1.0,  1.0,  1.0],
                                     [1.0,  1.0, -1.0,  1.0],
                                     [1.0, -1.0, -1.0, -1.0]]
        if problem == "AND": data = [[1.0,  1.0,  1.0,  1.0],
                                     [1.0, -1.0,  1.0, -1.0],
                                     [1.0,  1.0, -1.0, -1.0],
                                     [1.0, -1.0, -1.0, -1.0]]

        M = self.total_layers - 1                                   # M sera el indice de la capa superior
        nodes_per_layer = max(4, len(data[0]) - 1)                  # Cuantos nodos hay en las capas ocultas (incluye el del bias)
        exit_nodes = 1                                              # Cuantos nodos hay en la capa superior
        V = np.zeros((M+1, nodes_per_layer))                        # [capa, j]
        for i in range(1, M):
            V[i][0] = 1                                             # Bias para cada capa
        W = np.random.rand(M+1, nodes_per_layer, nodes_per_layer)   # [capa destino, dest, origen]
        w = np.random.rand(nodes_per_layer, len(data[0]) - 1)       # [dest, origen]
        W[1,:,:] = np.zeros((nodes_per_layer, nodes_per_layer))
        d = np.zeros((M+1, nodes_per_layer))
        for orig in range(len(data[0])-1):
            for dest in range(nodes_per_layer):
                W[1,dest,orig] = w[dest,orig]
        
        error_min = 20
        total_error = 1
        for epoch in range(1, self.iterations):
            total_error = 0
            # Randomize W every once in a while
            if (epoch % 100 == 99):
                W = np.random.rand(M+1, nodes_per_layer, nodes_per_layer)   # [capa destino, dest, origen]
                W[1,:,:] = np.zeros((nodes_per_layer, nodes_per_layer))
                for orig in range(len(data[0])-1):
                    for dest in range(nodes_per_layer):
                        W[1,dest,orig] = w[dest,orig]


            for mu in range(len(data)):
                # Paso 2 (V0 tiene los ejemplos iniciales)
                for k in range(len(data[0])-1):
                    V[0][k] = data[mu][k]
                
                # Paso 3A (Vi tiene los resultados de cada perceptron en la capa m)
                for m in range(1,M):
                    for i in range(1,nodes_per_layer):
                        hmi = self.h(m, i, nodes_per_layer, W, V)
                        V[m][i] = self.g(hmi)

                # Paso 3B (En la ultima capa habra exit_nodes en vez de nodes_per_layer)
                for i in range(0,exit_nodes):
                    hMi = self.h(M, i, nodes_per_layer, W, V)
                    V[M][i] = self.g(hMi)

                # Paso 4 (Calculo error para capa de salida M)
                for i in range(0,exit_nodes):
                    hMi = self.h(M, i, nodes_per_layer, W, V)
                    if exit_nodes == 1:
                        d[M][i] = self.g_derivative(hMi)*(data[mu][-1] - V[M][i])    
                    else:
                        d[M][i] = self.g_derivative(hMi)*(data[mu][-1][i] - V[M][i])

                # Paso 5 (Retropropagar error)
                for m in range(M,1,-1):                                             # m es la capa superior
                    for j in range(0,nodes_per_layer):                              # Por cada j en el medio
                        hprevmi = self.h(m-1, j, nodes_per_layer, W, V)             # hj = hj del medio
                        error_sum = 0
                        for i in range(0, nodes_per_layer):                         # Por cada nodo en la capa superior
                            error_sum += W[m,i,j] * d[m][i]                         # sumo la rama de aca hasta arriba y multiplico por el error
                        d[m-1][j] = self.g_derivative(hprevmi) * error_sum

                # Paso 6 (Actualizar pesos)
                for m in range(1,M+1):
                    for i in range(nodes_per_layer):
                        for j in range(nodes_per_layer):
                            delta = self.alpha * d[m][i] * V[m-1][j]
                            W[m,i,j] = W[m,i,j] + delta

                # Paso 7 (Calcular error)
                for i in range(0,exit_nodes):
                    if exit_nodes == 1:
                        total_error += abs(data[mu][3] - V[M][i])
                    else:
                        total_error += abs(data[mu][3][i] - V[M][i])
                if total_error < error_min:
                    error_min = total_error
                    w_min = W
                if total_error <= 0:
                    break
        print(V)
        self.create_plot(data, w_min[M,:,:], problem)
        return