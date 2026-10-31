# Mathematical Background of Gradient Descent

## 1. Objective

Gradient descent is an optimization algorithm used to find model parameters that minimize a loss function.

For simple linear regression, the model is:

$$
\hat{y}_i = b + wx_i
$$

where:

- $\hat{y}_i$ is the predicted value.
- $y_i$ is the actual value.
- $b$ is the intercept (bias).
- $w$ is the weight (slope).
- $x_i$ is the input feature.
- $n$ is the number of training examples.

The goal is to find the best values of $b$ and $w$.

## 2. Loss Function

The loss function measures the difference between predictions and actual values.

We use mean squared error with a factor of one-half:

$$
L(b,w) = \frac{1}{2n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

Substituting the linear regression model:

$$
L(b,w) = \frac{1}{2n}\sum_{i=1}^{n}(b+wx_i-y_i)^2
$$

The factor $\frac{1}{2}$ simplifies the derivatives because the factor of $2$ from differentiating the squared error cancels it.

Our objective is:

$$
\min_{b,w} L(b,w)
$$

## 3. Calculate the Gradient

The gradient tells us how the loss changes when we change each model parameter.

### 3.1 Derivative with respect to the intercept

$$
\frac{\partial L}{\partial b}
=
\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
$$

### 3.2 Derivative with respect to the weight

$$
\frac{\partial L}{\partial w}
=
\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)x_i
$$

### 3.3 Gradient vector

We can combine both derivatives into one vector:

$$
\nabla L(b,w)
=
\begin{bmatrix}
\frac{\partial L}{\partial b}\\[6pt]
\frac{\partial L}{\partial w}
\end{bmatrix}
$$

### 3.4 Matrix notation

We can express linear regression using matrix multiplication.

Include a column of ones in the design matrix to account for the intercept:

$$
X =
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix}
$$

Define the parameter and target vectors:

$$
\theta =
\begin{bmatrix}
b\\
w
\end{bmatrix},
\qquad
y =
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}
$$

Predictions are:

$$
\hat{y}=X\theta
$$

The loss is:

$$
L(\theta)=\frac{1}{2n}\|X\theta-y\|_2^2
$$

Its gradient is:

$$
\nabla L(\theta)=\frac{1}{n}X^T(X\theta-y)
$$

**Important:** Because the loss includes the factor $\frac{1}{2}$, the gradient does not contain a factor of $2$.

## 4. Gradient Descent Update Rule

Gradient descent updates parameters in the opposite direction of the gradient:

$$
\theta^{(t+1)}
=
\theta^{(t)}-\alpha\nabla L(\theta^{(t)})
$$

Here:

- $\theta^{(t)}$ represents the current parameters.
- $\theta^{(t+1)}$ represents the updated parameters.
- $\alpha$ is the learning rate.
- $\nabla L$ is the gradient.

The learning rate determines the size of each update.

- If the learning rate is too large, training may overshoot the minimum or diverge.
- If the learning rate is too small, training may be slow.

The individual parameter updates are:

$$
b_{\text{new}}
=
b_{\text{old}}-\alpha\frac{\partial L}{\partial b}
$$

$$
w_{\text{new}}
=
w_{\text{old}}-\alpha\frac{\partial L}{\partial w}
$$

## 5. Numerical Example

Consider the following training data:

| $x$ | $y$ |
|---:|---:|
| 1 | 5 |
| 2 | 8 |
| 3 | 11 |

These points follow the relationship:

$$
y=2+3x
$$

Therefore, the ideal intercept is $b=2$, and the ideal weight is $w=3$.

### Step 1: Construct the design matrix

$$
X=
\begin{bmatrix}
1 & 1\\
1 & 2\\
1 & 3
\end{bmatrix}
$$

The target vector is:

$$
y=
\begin{bmatrix}
5\\
8\\
11
\end{bmatrix}
$$

### Step 2: Initialize the parameters

Start with zero parameters:

$$
\theta^{(0)}
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

### Step 3: Calculate the initial predictions

$$
\hat{y}=X\theta^{(0)}
$$

Therefore:

$$
\hat{y}=
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
$$

### Step 4: Calculate the error

Define the error vector as $e=\hat{y}-y$.

$$
e=
\begin{bmatrix}
-5\\
-8\\
-11
\end{bmatrix}
$$

### Step 5: Calculate the gradient

Using the gradient formula:

$$
\nabla L(\theta^{(0)})
=
\frac{1}{3}X^Te
$$

First, calculate the transpose:

$$
X^T=
\begin{bmatrix}
1 & 1 & 1\\
1 & 2 & 3
\end{bmatrix}
$$

Multiply the matrices:

$$
X^Te=
\begin{bmatrix}
1 & 1 & 1\\
1 & 2 & 3
\end{bmatrix}
\begin{bmatrix}
-5\\
-8\\
-11
\end{bmatrix}
$$

This gives:

$$
X^Te=
\begin{bmatrix}
-24\\
-54
\end{bmatrix}
$$

Divide by $n=3$:

$$
\nabla L(\theta^{(0)})
=
\begin{bmatrix}
-8\\
-18
\end{bmatrix}
$$

Thus, the gradient of the intercept is $-8$, and the gradient of the weight is $-18$.

### Step 6: Update the parameters

Let the learning rate be $\alpha=0.01$.

$$
\theta^{(1)}
=
\theta^{(0)}-\alpha\nabla L(\theta^{(0)})
$$

Substitute the values:

$$
\theta^{(1)}
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
-
0.01
\begin{bmatrix}
-8\\
-18
\end{bmatrix}
$$

The updated parameters are:

$$
\boxed{
\theta^{(1)}
=
\begin{bmatrix}
0.08\\
0.18
\end{bmatrix}
}
$$

After just one update:

- The intercept changes from $0$ to $0.08$.
- The weight changes from $0$ to $0.18$.

These values are not yet optimal. Gradient descent continues updating them until the loss becomes sufficiently small.

## 6. Python Implementation Using NumPy

The following code implements gradient descent manually.

```python
import numpy as np

# Training data
X = np.array([
    [1.0, 1.0],
    [1.0, 2.0],
    [1.0, 3.0]
])

y = np.array([5.0, 8.0, 11.0])

# Initialize parameters: [intercept, weight]
theta = np.zeros(2)

learning_rate = 0.01
epochs = 1000
n = len(y)

# Initial loss
predictions = X @ theta
loss = np.sum((predictions - y) ** 2) / (2 * n)

print(f"Initial loss: {loss:.6f}")

for epoch in range(epochs):
    # Forward pass
    predictions = X @ theta

    # Calculate error
    error = predictions - y

    # Calculate gradient
    gradient = (X.T @ error) / n

    # Update parameters
    theta = theta - learning_rate * gradient

    if (epoch + 1) % 100 == 0:
        predictions = X @ theta
        loss = np.sum((predictions - y) ** 2) / (2 * n)
        print(f"Epoch {epoch + 1}: Loss = {loss:.6f}")

print("\nLearned intercept:", theta[0])
print("Learned weight:", theta[1])
```

The learned intercept should approach $2$, and the learned weight should approach $3$.

## 7. Batch, Stochastic, and Mini-Batch Gradient Descent

### Batch Gradient Descent

Uses the entire training dataset to calculate each gradient.

$$
\theta\leftarrow\theta-\alpha\frac{1}{n}X^T(X\theta-y)
$$

- One update uses all $n$ training examples.
- The gradient is generally stable.
- Large datasets can make each update expensive.

### Stochastic Gradient Descent

Uses one training example for each update.

For an individual example with augmented feature vector $x_i$:

$$
\theta\leftarrow\theta-\alpha x_i(x_i^T\theta-y_i)
$$

- Updates parameters after every example.
- Each update is inexpensive.
- The optimization path is noisier.

### Mini-Batch Gradient Descent

Uses a small batch of $m$ training examples.

Let $X_B$ be the batch design matrix and $y_B$ its target vector.

$$
\theta\leftarrow\theta-\alpha\frac{1}{m}X_B^T(X_B\theta-y_B)
$$

- Balances computational efficiency and gradient stability.
- Commonly used for large machine-learning datasets.
- PyTorch supports mini-batch training through `DataLoader`.

## 8. Closed-Form Solution

For ordinary least-squares linear regression, we can also calculate the solution directly using the normal equation:

$$
\theta^*=(X^TX)^{-1}X^Ty
$$

This formula requires $X^TX$ to be invertible. In practice, a least-squares solver is often safer than explicitly calculating the inverse.

For example:

```python
theta_solution = np.linalg.lstsq(X, y, rcond=None)[0]

print("Intercept:", theta_solution[0])
print("Weight:", theta_solution[1])
```

For the example dataset, the solution is:

$$
\boxed{
\theta^*=
\begin{bmatrix}
2\\
3
\end{bmatrix}
}
$$

## 9. Key Takeaways

1. Linear regression predicts values using a weighted sum of input features.
2. The loss function measures prediction error.
3. The gradient indicates how the loss changes with the parameters.
4. Gradient descent updates parameters in the opposite direction of the gradient.
5. The learning rate controls the size of each update.
6. Batch, stochastic, and mini-batch gradient descent differ in how much data they use per update.
7. The normal equation provides a direct least-squares solution when its assumptions hold.
8. PyTorch autograd can calculate gradients automatically, reducing the need to derive and implement them manually.

