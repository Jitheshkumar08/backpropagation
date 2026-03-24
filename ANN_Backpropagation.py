import numpy as np

# Sigmoid activation function → converts value between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid → used for learning (backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Input and expected output
X = np.array([[1]])   # input value
y = np.array([[1]])   # target value

# Initialize weight and bias
W = np.array([[0.5]])  # starting weight
b = 0                  # bias
lr = 0.1               # learning rate

print("Initial Weight:", W)

# Output:
# [[0.5]]

# Forward propagation → calculate prediction
z = np.dot(X, W) + b
y_pred = sigmoid(z)

print("Predicted Output:", y_pred)

# Output:
# [[0.62245933]]

# Explanation:
# sigmoid(0.5) = 0.622 → predicted value

# Loss calculation (MSE) → measures error
loss = 0.5 * (y - y_pred) ** 2
print("Loss:", loss)

# Output:
# [[0.0713]]  (approx)

# Explanation:
# Error = (1 - 0.622)^2 / 2 ≈ 0.0713

# Backpropagation → find gradient (error direction)
dL_dy = -(y - y_pred)
dy_dz = sigmoid_derivative(y_pred)
grad = np.dot(X.T, dL_dy * dy_dz)

# Intermediate values:
# dL/dy ≈ -0.3775
# dy/dz ≈ 0.2350
# grad ≈ -0.0887

# Update weight → reduce error
W = W - lr * grad

print("Updated Weight:", W)

# Output:
# [[0.50887]]  (approx)