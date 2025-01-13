from redes_neuronales.neuron import Neuron

def logic_not(x):
    if x >= 0:
        return 1
    else:
        return 0

n1 = Neuron(x = [0], weights=[1], bias=100, func="Binary Step")
print(logic_not(n1.predict()))
n2 = Neuron(x = [-21], weights=[-2], bias=-43, func="Binary Step")
print(logic_not(n2.predict()))