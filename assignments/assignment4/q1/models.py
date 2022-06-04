import numpy as np


class Perceptron:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)

    def __call__(self, x):
        return x.dot(self.W) + self.b


