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

if perceptron != 'SIMPLE' and perceptron != 'MULTI':
    exit('Error with perceptron selection. Possible values: "SIMPLE" or "MULTI"')
elif perceptron == 'SIMPLE' and (operand != 'AND' and operand != 'XOR' and operand != 'Ej2'):
    exit('Error with problem selection. Possible values for simple perceptron: "XOR" or "AND" or "Ej2"')
elif perceptron == 'MULTI' and (operand != 'XOR' and operand != 'EVEN'):
    exit('Error with problem selection. Possible values for multi layer perceptron: "XOR" or "EVEN"')
if alpha < 0.0:
    exit('Error with learning rate. Value must be positive')
if adaptive != True and adaptive != False:
    exit('Error with adaptive learning rate option. Possible values are: "TRUE" or "FALSE"')
if epochs < 0.0:
    exit('Error with epochs. Value must be positive')
if beta < 0.0:
    exit('Error with beta coefficient. Value must be positive')
if error_tolerance < 0.0:
    exit('Error with error tolerance. Value must be positive')
if classification_margin < 0.0:
    exit('Error with classification margin. Value must be positive')
if hidden_layers < 0.0:
    exit('Error with hidden layers. Value must be positive')
if nodes_per_layer < 0.0:
    exit('Error with nodes per layer. Value must be positive')

if(perceptron == 'SIMPLE'):
    if(operand == 'Ej2'):
        p = SimplePerceptronEj2(alpha=alpha, iterations=epochs, beta=beta, adaptive=adaptive)
    else:
        p = SimplePerceptron(alpha=alpha, iterations=epochs, adaptive=adaptive)
else:
	p = MultiLayerPerceptron(alpha=alpha, beta=beta, iterations=epochs, hidden_layers=hidden_layers, nodes_per_layer=nodes_per_layer,
			error_tolerance=error_tolerance, adaptive=adaptive, classification_margin=classification_margin)

p.algorithm(operand)


