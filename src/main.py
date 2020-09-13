from perceptron import SimplePerceptron
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../')
import json
with open(os.getcwd() + "/input.json") as file:
    operand = json.load(file)['FUNCTION']

p = SimplePerceptron(0.5, 100)
p.algorithm(operand)

