"""
Neural Network Optimizers from Scratch (NumPy).
Covers: SGD, Momentum, RMSProp, and Adam.

Math (Adam):
- m_t = β1 * m_{t-1} + (1 - β1) * g_t (First moment)
- v_t = β2 * v_{t-1} + (1 - β2) * g_t^2 (Second moment)
- θ_{t+1} = θ_t - η * m_t / (sqrt(v_t) + ε)
"""

import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update weights
        w_new = w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w_new

if __name__ == "__main__":
    # Toy quadratic optimization problem: minimize f(x) = x^2
    # Gradient: f'(x) = 2x
    x = 10.0
    optimizer = AdamOptimizer(learning_rate=0.1)

    print("Optimizing f(x) = x^2 starting at x = 10")
    for i in range(50):
        grad = 2 * x
        x = optimizer.update(x, grad)
        if i % 5 == 0:
            print(f"Iter {i}, x: {x:.6f}, f(x): {x**2:.6f}")
