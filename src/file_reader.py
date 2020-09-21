import numpy as np
import string
 
class Reader:
 
    def __init__(self, excercise):
        self.excercise = excercise
    
    def readFile(self, test=False):
        if test == True:
            return self.readExcerciseTwo(10)
        if self.excercise == 'Ej2':
            return self.readExcerciseTwo()
        if self.excercise == 'Ej3':
            return self.readExerciseThree(5, 7)

    def readExcerciseTwo(self, amount=50):
        f = open('TP3-ej2-Conjunto-entrenamiento.txt', 'r')
        g = open('TP3-ej2-Salida-deseada.txt', 'r')
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        valuesg = self.normalize_output([float(line.strip()) for line in linesg])
        values_indiv = [line.split(' ') for line in valuesf]
        idx = 0
        nice = []
        full_data_row_count = len(valuesg)
        for count in range(amount):
            random_row = random.randint(0, full_data_row_count - 1)
            nice.append([1.0])
            row = values_indiv[random_row]
            for element in row:
                if element != '':
                    nice[count].append(float(element))
            nice[count].append(float(valuesg[random_row]))
        return nice

    def readExerciseThree(self, width, height):
        f = open('TP3-ej3-mapa-de-pixeles-digitos-decimales.txt', 'r')
        linesf = f.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        values_indiv = [line.split(' ') for line in valuesf]
        data = np.zeros((10, width*height + 2))
        for index in range(10):                         # 10 veces
            for fila in range(height):                  # 7 veces
                for col in range(width):                # 5 veces
                    data[index][1+col+fila*width] = string.atoi(values_indiv[index*height+fila][col])
        for i in range(len(data)):
            data[i][0] = 1
            data[i][-1] = (i%2 * 2) - 1
        return data
        
    def normalize_output(self, outputs): # tengo que normalizar sino está fuera del rango de activacion de la tanh o sigmoidea
        max_output = np.max(outputs)
        min_output = np.min(outputs)
        for i in range(0, len(outputs)):
            outputs[i] =  (outputs[i] - min_output) / (max_output - min_output)
        return outputs
