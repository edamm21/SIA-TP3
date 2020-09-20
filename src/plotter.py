from matplotlib import pyplot as plt 
import numpy as np

class Plotter:

    def create_plot_ej1(self, data, weights, operand):
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