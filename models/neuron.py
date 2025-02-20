from models.value import Value
import random

class Neuron:
    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))
        
    def __call__(self, inputs):
        act = sum((wi * xi for wi, xi in zip(self.weights, inputs)), self.bias)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.weights + [self.bias]