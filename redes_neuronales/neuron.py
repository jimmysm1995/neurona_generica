import numpy as np

class Neuron:
    def __init__(self, x, weights, bias, func):
        self.x = x
        self.weights = weights
        self.bias = bias
        self.func = func

    def predict(self):
        total = sum(w * x for w, x in zip(self.weights, self.x)) + self.bias
        if self.func == "ReLU":
            return self.__relu(total)
        if self.func == "Sigmoide":
            return self.__sigmoid(total)
        if self.func == "Tangente hiperbólica":
            return self.__tanh(total)
        if self.func == "Binary Step":
            return self.__binary_step(total)
        return total

    def logic_not(self):
        return self.predict() > 0.5

    def change_bias(self, bias):
        self.bias = bias

    def __relu(self, x):
        return max(0, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, total):
        return np.tanh(total)

    def __binary_step(self, x):
        return 1 if x >= 0 else 0