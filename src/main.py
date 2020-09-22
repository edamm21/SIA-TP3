from file_reader import Reader
from ej1 import SimplePerceptron
from ej2 import SimplePerceptronEj2
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
    adaptive = data['ADAPTIVE_LEARNING_RATE']

if(perceptron == 'SIMPLE'):
    adaptive = adaptive == 'TRUE'
    if(operand == 'Ej2'):
        p = SimplePerceptronEj2(alpha=alpha, iterations=epochs, beta=beta, adaptive=adaptive)
    else:
        p = SimplePerceptron(alpha=alpha, iterations=epochs, adaptive=adaptive)
else:
    p = MultiLayerPerceptron(alpha=alpha, iterations=epochs, hidden_layers=1, error_tolerance=error_tolerance, adaptive=True)

p.algorithm(operand)


