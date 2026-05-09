"""
Scaled Dot-Product Attention from Scratch (NumPy).
The fundamental block of the Transformer architecture.

Math:
Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
"""

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Computes Attention.
    q, k, v are of shape (seq_len, d_k)
    """
    d_k = q.shape[-1]
    
    # 1. Compute Scores (Q * K^T)
    scores = np.matmul(q, k.T) / np.sqrt(d_k)
    
    # 2. Apply Mask (optional - for causal/padding)
    if mask is not None:
        scores += (mask * -1e9)
        
    # 3. Apply Softmax to get weights
    weights = softmax(scores)
    
    # 4. Multiply by V
    output = np.matmul(weights, v)
    
    return output, weights

if __name__ == "__main__":
    # Toy example: 3 words, each embedding size 4
    seq_len = 3
    d_k = 4
    
    # Random Query, Key, Value
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("Query Matrix Shape:", Q.shape)
    print("Attention Weights (Softmax output):")
    print(weights)
    print("\nAttention Output Shape:", output.shape)
