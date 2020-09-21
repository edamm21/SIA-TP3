import numpy as np
import string
 
class Reader:
 
    def __init__(self, excercise):
        self.excercise = excercise
    
    def readFile(self):
        if self.excercise == 'Ej2':
            return self.readExcerciseTwo()
        if self.excercise == 'Ej3':
            return self.readExerciseThree(5, 7)
        if self.excercise == 'TEST':
            return self.readTest()
 
    def readTest(self):
        f = open('test.txt', 'r')
        g = open('test-out.txt', 'r')
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        valuesg = self.normalize_output([float(line.strip()) for line in linesg])
        values_indiv = [line.split(' ') for line in valuesf]
        idx = 0
        nice = []
        for row in values_indiv:
            nice.append([1.0])
            for element in row:
                if element != '':
                    nice[idx].append(float(element))
            nice[idx].append(float(valuesg[idx]))
            idx += 1
        return nice
 
    def readExcerciseTwo(self):
        f = open('TP3-ej2-Conjunto-entrenamiento.txt', 'r')
        g = open('TP3-ej2-Salida-deseada.txt', 'r')
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        valuesg = self.normalize_output([float(line.strip()) for line in linesg])
        values_indiv = [line.split(' ') for line in valuesf]
        idx = 0

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