"""
Multi-Layer Perceptron (MLP) from Scratch with NumPy.
Implements forward pass, backpropagation, and weight updates.

Math:
- Forward: a = σ(w * x + b)
- Backprop (Chain Rule): ∂L/∂w = (∂L/∂a) * (∂a/∂z) * (∂z/∂w)
"""

import numpy as np

class MLPFromScratch:
    def __init__(self, layers):
        self.weights = [np.random.randn(layers[i], layers[i+1]) * 0.01 for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)
            self.activations.append(self.sigmoid(z))
        return self.activations[-1]

    def backward(self, X, y, output, learning_rate):
        error = y - output
        delta = error * self.sigmoid_derivative(output)

        for i in reversed(range(len(self.weights))):
            current_activation = self.activations[i]
            self.weights[i] += learning_rate * np.dot(current_activation.T, delta)
            self.biases[i] += learning_rate * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])

    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, lr)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

if __name__ == "__main__":
    # XOR Problem Demo
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    model = MLPFromScratch([2, 4, 1])
    print("Training MLP on XOR problem...")
    model.train(X, y, epochs=10000, lr=0.1)

    print("\nPredictions:")
    print(model.forward(X))
