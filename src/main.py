from file_reader import Reader
from single_layer_perceptron import SimplePerceptron
from multi_layer_perceptron import MultiLayerPerceptron
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../')
import json
with open(os.getcwd() + "/input.json") as file:
    data = json.load(file)
    operand = data['FUNCTION']
    alpha = data['LEARNING_RATE']
    epochs = data['EPOCHS']
    perceptron = data['PERCEPTRON']
    beta = data['BETA']
    error_tolerance = data['ERROR_TOLERANCE']


if(perceptron == 'SIMPLE'):
    if(operand == 'Ej2'):
        p = SimplePerceptron(alpha=alpha, iterations=epochs, linear=False, beta=beta)
    else:
        p = SimplePerceptron(alpha=alpha, iterations=epochs, linear=True, beta=beta)
    p.algorithm(operand)
else:
    p = MultiLayerPerceptron(alpha=alpha, iterations=epochs, hidden_layers=1, error_tolerance=error_tolerance)
    p.algorithm(operand)

