# Notes

Gradient descent for neural networks:

1.  Gradient descent
2.  forward propagation
3.  backward propagation
4.

The general methodology to build a Neural Network:

1. Define the neural network structure ( # of input units, $ of hidden units, etc)
2. Initialize the model's parameters
3. Loop:
   3.1 Implement forward propagation
   3.2 Compute loss
   3.3 Implement backward propagation to get the gradients
   3.4 Update parameters(gradient descent)

## Backpropagation

Backpropagation is essentially the application of the **Chain Rule** from calculus to calculate the gradient of the loss function with respect to every weight and bias in the network.

Here is the mathematical derivation and a clean NumPy implementation for a single-layer network.

---

## 1. The Math Formula

We want to find how much the Loss () changes when we change a weight ().

Assuming:

- $z = w \cdot a_{prev} + b$
- (Activation)
- is the Loss function.

The Chain Rule tells us:

### The Four Fundamental Equations:

1. **Error in output layer ():** For Cross-Entropy with Sigmoid, this simplifies beautifully to:

2. **Gradient of Weights ():**

3. **Gradient of Bias ():**

4. **Error in previous layer ():** Used to pass the gradient further back:

---

## 2. NumPy Implementation

This implementation shows one "backward pass" for a network with one hidden layer.

```python
import numpy as np

def backward_propagation(parameters, cache, X, Y):
    """
    parameters: dictionary containing weights W1, W2 and biases b1, b2
    cache: dictionary containing Z1, A1, Z2, A2 from forward pass
    X: input data
    Y: true labels
    """
    m = X.shape[1] # number of examples

    # Retrieve from cache
    A1 = cache['A1']
    A2 = cache['A2']
    W2 = parameters['W2']

    # --- Output Layer (Layer 2) ---
    # 1. Calculate dz2 (Error at output)
    dz2 = A2 - Y

    # 2. Calculate dW2 and db2
    dW2 = (1/m) * np.dot(dz2, A1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    # --- Hidden Layer (Layer 1) ---
    # 1. Calculate dz1 (Backpropagate error through W2 and activation)
    # Note: derivative of tanh is (1 - A^2)
    dz1 = np.dot(W2.T, dz2) * (1 - np.power(A1, 2))

    # 2. Calculate dW1 and db1
    dW1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

```

---

## 3. Key Concepts to Remember

- **Matrix Dimensions:** This is where most errors happen. If is and is , you must transpose to get a of .
- **The Activation Derivative:** Notice the `(1 - np.power(A1, 2))` in the hidden layer calculation. That is the derivative of the **tanh** function. If you used **ReLU**, you would use a mask where .
- **The Learning Step:** These `grads` are then used to update the parameters:

---

### Summary Table: Forward vs. Backward

| Step           | Forward Pass        | Backward Pass (Gradients) |
| -------------- | ------------------- | ------------------------- |
| **Linear**     |                     |                           |
| **Activation** |                     |                           |
| **Goal**       | Generate Prediction | Update Parameters         |

**Would you like me to show the derivative for the ReLU or GeLU activation functions to replace the tanh used above?**
