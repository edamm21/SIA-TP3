import numpy as np
import string
import random

class Reader:
 
    def __init__(self, excercise):
        self.excercise = excercise
        self.readContent = []
    
    def readFile(self, test=False):
        if self.excercise == 'Ej2' and test == True:
            return self.readExcerciseTwo(10)
        if self.excercise == 'Ej2' and test != True:
            return self.readExcerciseTwo()
        if self.excercise == 'Ej3' and test == True:
            X = self.readExerciseThree(5, 7, 10, test).tolist()
            Y = self.readContent.tolist()
            for elem in Y:
                X.remove(elem)
            return X
        if self.excercise == 'Ej3' and test != True:
            return self.readExerciseThree(5, 7, 7, test)

    def readExcerciseTwo(self, amount=180):
        f = open('TP3-ej2-Conjunto-entrenamiento.txt', 'r')
        g = open('TP3-ej2-Salida-deseada.txt', 'r')
        linesf = f.read().split('\n')
        linesg = g.read().split('\n')
        total_amount = len(linesf)
        testing_amount = total_amount - amount
        valuesf = [line.strip() for line in linesf]
        valuesg_normalized, max_out, min_out = self.normalize_output([float(line.strip()) for line in linesg])
        valuesg = [float(line.strip()) for line in linesg]
        values_indiv = [line.split(' ') for line in valuesf]
        idx = 0
        nice = []
        nice_normalized = []
        full_data_row_count = len(valuesg)
        training_indexes = set()
        while len(training_indexes) < amount:
            training_indexes.add(random.randint(0, full_data_row_count - 1))
        training_indexes_list = list(training_indexes)
        for count in range(amount):
            nice.append([1.0])
            nice_normalized.append([1.0])
            row = values_indiv[training_indexes_list[count]]
            for element in row:
                if element != '':
                    nice[count].append(float(element))
                    nice_normalized[count].append(float(element))
            nice[count].append(float(valuesg[training_indexes_list[count]]))
            nice_normalized[count].append(float(valuesg_normalized[training_indexes_list[count]]))
        testing_indexes_list = list(set(range(0, total_amount)) - training_indexes)
        test = []
        test_normalized = []
        for count in range(testing_amount):
            test.append([1.0])
            test_normalized.append([1.0])
            row = values_indiv[testing_indexes_list[count]]
            for element in row:
                if element != '':
                    test[count].append(float(element))
                    test_normalized[count].append(float(element))
            test[count].append(float(valuesg[testing_indexes_list[count]]))
            test_normalized[count].append(float(valuesg_normalized[testing_indexes_list[count]]))
        return nice, nice_normalized, test, test_normalized, max_out, min_out

    def readExerciseThree(self, width=5, height=7, amount=10, test=False):
        f = open('TP3-ej3-mapa-de-pixeles-digitos-decimales.txt', 'r')
        linesf = f.read().split('\n')
        valuesf = [line.strip() for line in linesf]
        values_indiv = [line.split(' ') for line in valuesf]
        data = np.zeros((10, width*height + 2))
        data2 = np.zeros((amount, width*height + 2))
        for index in range(10):                     # Complete with the full thing
            for fila in range(height):
                for col in range(width):
                    data[index][1+col+fila*width] = int(values_indiv[index*height+fila][col])
        for i in range(len(data)):
            data[i][0] = 1
            data[i][-1] = (i%2 * 2) - 1
        np.random.shuffle(data)
        data2 = data[0:amount]
        if not test:
            self.readContent = data2
        return data2
        
    def normalize_output(self, outputs): # tengo que normalizar sino esta fuera del rango de activacion de la tanh o sigmoidea
        max_output = np.max(outputs)
        min_output = np.min(outputs)
        for i in range(0, len(outputs)):
            outputs[i] =  (outputs[i] - min_output) / (max_output - min_output)
        return outputs, max_output, min_output