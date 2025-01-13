import neuron
from redes_neuronales.neuron import Neuron

n1 = Neuron(x = [4, 6, -3], weights=[0.5, 1.2, 6.8], bias=100, func="ReLU")
output = n1.predict()
print(output)
n1.change_bias(-100)
output = n1.predict()
print(output)