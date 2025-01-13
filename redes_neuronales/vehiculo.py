import neuron
from redes_neuronales.neuron import Neuron

output = float(input("Introduce el número de horas transcurridas(puede contener decimales):"))
n1 = Neuron(x = [output], weights=[100], bias=60, func="ReLU")
print("El vehículo ha recorrido", n1.predict(), "kilómetros")
output = float(input("Introduce el número de horas transcurridas(puede contener decimales):"))
n2 = Neuron(x = [output], weights=[100], bias=60, func="ReLU")
print("El vehículo ha recorrido", n2.predict(), "kilómetros")
output = float(input("Introduce el número de horas transcurridas(puede contener decimales):"))
n3 = Neuron(x = [output], weights=[100], bias=60, func="ReLU")
print("El vehículo ha recorrido", n3.predict(), "kilómetros")