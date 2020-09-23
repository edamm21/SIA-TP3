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
    classification_margin = data['CLASSIFICATION_MARGIN']
    hidden_layers = data['HIDDEN_LAYERS']
    nodes_per_layer = data['NODES_PER_LAYER']
    adaptive = adaptive == 'TRUE'
if(perceptron == 'SIMPLE'):
    if(operand == 'Ej2'):
        p = SimplePerceptronEj2(alpha=alpha, iterations=epochs, beta=beta, adaptive=adaptive)
    else:
        p = SimplePerceptron(alpha=alpha, iterations=epochs, adaptive=adaptive)
else:
	p = MultiLayerPerceptron(alpha=alpha, beta=beta, iterations=epochs, hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer,
			error_tolerance=error_tolerance, adaptive=adaptive, classification_margin=classification_margin)

p.algorithm(operand)


