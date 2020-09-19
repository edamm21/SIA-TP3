from file_reader import Reader
from ej1_single_layer_perceptron import SimplePerceptron
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

if(perceptron == 'SIMPLE'):
    p = SimplePerceptron(alpha=alpha, iterations=epochs, linear=False, beta=0.1)
    p.algorithm(operand)
else:
    p = MultiLayerPerceptron(alpha=alpha, iterations=epochs, hidden_layers=1)
    p.algorithm("EVEN")

