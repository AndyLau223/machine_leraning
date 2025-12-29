# notes

## Main Types of Machine Learning

- Supervised Learning - based on labeled data
  - Regression -> predict a number
  - Classification -> predict categories/classes
- Unsupervised Learning - model finds structures itself on the data
  - Clustering
  - Dimensionality Reduction
  - Association Rule Mining
  - Density Estimation
- Semi-Supervised Learning - small amount of labeled data + large amount of unlabeled data
  - Self-Training
  - Co-Training
  - Label Propagation
- Self-Supervised Learning
- Reinforcement Learning


## Regression


**Cost function**

---

Given linear model: $f_{w,b}(x)=wx+b$

parameters: w,b

Cost function: Squared error cost function

$$
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}\big(f_{w,b}(x^{(i)})-y^{(i)}\big)^2
$$

Or vectorized form:

$$
J(\theta)=\frac{1}{2m}(X\theta - y)^\top (X\theta - y)
$$

goal: $minimize_{w.b} J(w,b)$

- The cost function (J) measures the difference between predicted values and actual values in a dataset.
- The goal is to minimize (J) by finding optimal parameters (w) and (b) for the model

Why Square the error?

- removes negative sign
- Penalizes big mistakes more


**Gradient descent**

___




**Mutilple Features**

---


- vectorization
-


## Classification

Classification is a type of machine learning which predict cateogry.


**Vocabulary**

___

- Univariate linear regression
- Cost function
- mean normalization
- leanring curve, convergence
- feature engineering
- polynomial regression
- classification
- logistic regression
- sigmoid function/logistic function
- Log Loss (Cross-Entropy)
- Decision boundary 
- Non-linear decision boundaries 
- Cost function for logistic regression
- Gradient descent for logistic regression
- overfitting
- quadratic feature
- high bias
- high variance -> this is when the model does not generalize well to new exmaples
- all features + insufficient data == overfit 
- regularization 
- addressing overfitting 
  - collect more data
  - select features
  - reduce size of parameters 
    - regularization