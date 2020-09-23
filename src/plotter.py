from matplotlib import pyplot as plt
import matplotlib
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

        plt.scatter(negatives[0], negatives[1], s = 40.0, c = 'r', label = 'Proyeccion w < 0')
        plt.scatter(positives[0], positives[1], s = 40.0, c = 'b', label = 'Proyeccion w > 0')

        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()
    
    def create_plot_ej2(self, errors, test_errors, alphas, linear):
        fig,ax = plt.subplots()
        ax.set_title('Evolucion de error por epoca - {}'.format('LINEAL' if linear else 'NO LINEAL'))
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Error")
        epochs = []
        for i in range(len(errors)):
            if linear: alphas[i] *= 50000
            else: alphas[i] *= 10000
            epochs.append(i)
        plt.plot(epochs, errors, label='Errores de entrenamiento')
        plt.plot(epochs, test_errors, label='Errores de testeo')
        plt.plot(epochs, alphas, label='Variación del aprendizaje')
        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()

    def create_plot_ej3(self, errors, worst_errors, test_errors, test_worst_errors):
        fig,ax = plt.subplots()
        ax.set_title('Evolucion de error por epoca')
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Error")
        epochs = []
        for i in range(len(errors)):
            epochs.append(i)
        plt.plot(epochs, errors, 'cornflowerblue', label='Avg Training error')
        plt.plot(epochs, worst_errors, '-b', label='Max Training Error')
        plt.plot(epochs, test_errors, 'darkorange', label='Avg Test Error')
        plt.plot(epochs, test_worst_errors, '-r', label='Max Test Error')
        #'-g', label='Hiperplano')
        plt.legend(fontsize = 8)
        plt.grid(True)
        plt.show()

    def create_plot_ej3_accuracy(self, accuracy, test_accuracy):
        fig,ax = plt.subplots()
        ax.set_title('Evolucion de accuracy por epoca')
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Accuracy")
        epochs = []
        for i in range(len(accuracy)):
            epochs.append(i)
        plt.plot(epochs, accuracy, label='Training Accuracy')
        plt.plot(epochs, test_accuracy, label='Test Accuracy')
        plt.legend(fontsize = 8, loc = 0)
        plt.grid(True)
        plt.show()
