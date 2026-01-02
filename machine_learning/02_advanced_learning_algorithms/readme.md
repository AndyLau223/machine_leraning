# Advanced Learning Algorithms

---

**History of Neural Networks**

Neural networks were initially inspired by the human brain's learning processes.
They gained popularity in the 1980s, fell out of favor, and saw a resurgence around 2005 with the rise of deep learning.
---
**Biological Neurons vs. Artificial Neurons**

Biological neurons transmit electrical impulses and form connections, while artificial neurons perform computations on numerical inputs.
The analogy between biological and artificial neurons is loose, as current understanding of the brain is limited.

---

**Impact of Data and Technology**

The explosion of digital data and advancements in hardware, particularly GPUs, have enabled the success of deep learning.
Traditional algorithms struggled to scale with data, while larger neural networks showed improved performance with increased data availability.

---

**Neural Network Basics**

The example illustrates predicting whether a T-shirt will be a top seller based on features like price and sales data.
Logistic regression is introduced as a simplified model of a neuron, where the output represents the probability of a T-shirt being a top seller.

---

**Structure of Neural Networks**

A neural network consists of layers of neurons, where each layer processes inputs and produces outputs.
The input layer receives features, the hidden layer processes these features, and the output layer provides the final prediction.

---

**Feature Learning**

Neural networks can learn to create their own features, improving prediction accuracy without manual feature engineering.
The architecture of a neural network, including the number of hidden layers and neurons, significantly impacts its performance.

input layer -> hidden layer(s) -> output layer

---

**Understanding Neurons in a Layer**

A layer of neurons is a fundamental component of neural networks, where each neuron processes input features using logistic regression.
Each neuron has parameters (weights and biases) that determine its output activation value, calculated using the logistic function.

---

**Layer Structure and Notation**

Layers are indexed for clarity; for example, layer 1 is denoted with superscript [1], and layer 2 with [2].
The output from one layer serves as the input to the next, allowing for complex computations across multiple layers.

---

**Final Output and Predictions**

The final output layer computes a single prediction based on the activations from the previous layer, which can be thresholded to make binary predictions (e.g., yes or no).
This process illustrates how neural networks transform input data through layers to produce meaningful predictions.


---

**Neural Network Structure**

The example neural network consists of four layers: three hidden layers (Layers 1, 2, and 3) and one output layer (Layer 4), with the input layer referred to as Layer 0.
The total count of layers includes all hidden layers and the output layer, excluding the input layer.

---

Layer 3 Computations

Layer 3 receives a vector input from Layer 2 and outputs another vector, denoted as a_3.
The computations involve parameters (weights and biases) for each neuron in Layer 3, applying the sigmoid activation function to calculate the activations.

---

General Activation Function Notation

The general form for computing activations in any layer is presented, emphasizing the use of weights, biases, and the output from the previous layer.
The activation function, denoted as g, is identified as the sigmoid function, which outputs activation values for the neurons.

---

### **Activation Functions (Formulas)** ✅

**Sigmoid (Logistic)**  
- Markdown: `$ \sigma(x) = \dfrac{1}{1 + e^{-x}} $`  
- Plain: `sigma(x) = 1 / (1 + e^{-x})`  
- Derivative: `$ \sigma'(x) = \sigma(x)\,(1 - \sigma(x)) $`

---

**Tanh (Hyperbolic tangent)**  
- Markdown: `$ \tanh(x) = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}} $`  
- Plain: `tanh(x) = (e^x - e^-x) / (e^x + e^-x)`  
- Derivative: `$ \tanh'(x) = 1 - \tanh^2(x) $`

---

**ReLU (Rectified Linear Unit)** ⚡  
- Markdown: `$ \mathrm{ReLU}(x) = \max(0, x) $`  
- Plain: `ReLU(x) = max(0, x)`  
- Derivative: `$ \mathrm{ReLU}'(x) = \begin{cases} 1 & x>0 \\ 0 & x\le 0 \end{cases} $`

---

**Leaky ReLU** 🔧  
- Markdown: `$ \mathrm{LeakyReLU}(x) = \max(\alpha x, x) $` (α small, e.g., 0.01)  
- Plain: `LeakyReLU(x) = max(alpha * x, x)`  
- Derivative: `$ \mathrm{LeakyReLU}'(x) = \begin{cases} 1 & x>0 \\ \alpha & x\le 0 \end{cases} $`

---

**Softmax (vector form)** 🎯  
- Markdown: `$ \mathrm{softmax}(z)_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}} $`  
- Plain: `softmax(z)_i = exp(z_i) / sum_j exp(z_j)`

---

**Linear (Identity)** 🔁  
- Markdown: `$ f(x) = x $`  
- Plain: `f(x) = x`  
- Derivative: `$ f'(x) = 1 $`

---

**Swish** ✨  
- Markdown: `$ \mathrm{swish}(x) = x \cdot \sigma(x) $`  
- Plain: `swish(x) = x * sigmoid(x)`

---

> **Note:** Use **Sigmoid** for binary outputs, **Softmax** for multiclass, and **ReLU/LeakyReLU** for hidden layers. ✅


**Training a Neural Network in TensorFlow**

Step 1: Specify Output Function

Define how to compute the output given input features and model parameters.
This involves setting up the architecture of the neural network, including the number of hidden layers and units.

Step 2: Define Loss and Cost Functions

Specify the loss function, such as binary cross-entropy for classification tasks.
The cost function is the average of the loss over all training examples, guiding the optimization process.

Step 3: Minimize Cost Function
Use gradient descent to minimize the cost function by updating model parameters iteratively.
TensorFlow automates this process through backpropagation, allowing for efficient training of the neural network.

---

Vocabulary:

- neurons
- nucleus
- dendrites
- axons
- input layer
- hidden layer
- output layer
- multiclass classification
- activation function
- Choosing activation functions
- softmax regression algorithm
- training set/test set
- train accuracy/test accuracy
- model selection
- training/cross validation/test sets
- training/cross validation/test error
- generalization error
 - Anverage error on new examples not in the training set
- what is the purpsoe of model selection?
 - To choose the best model architecture and hyperparameters that minimize generalization error

