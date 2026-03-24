


#RNN is a type of neural network used for sequential data like text, speech, or time series.

import numpy as np

# Initialize weights → random values for input and hidden connections
Wx = 0.5    # (example value instead of random for explanation)
Wh = 0.3

print("Initial Wx:", Wx)
print("Initial Wh:", Wh)

# Output:
# Initial Wx: 0.5
# Initial Wh: 0.3

# Initial hidden state → starts with zero (no memory)
h_prev = 0

# Input sequence → sequential data (like time steps)
inputs = [1, 2, 3]

hidden_states = []

# Forward pass → process each input step by step
for x in inputs:
    # Combine current input and previous hidden state
    z = Wx * x + Wh * h_prev
    
    # Apply activation
    h = np.tanh(z)
    
    print("\nInput:", x)
    print("z (Wx*x + Wh*h_prev):", z)
    print("Hidden state:", h)
    
    # Store hidden state
    hidden_states.append(h)
    
    # Update memory
    h_prev = h

print("\nHidden States:")
print(hidden_states)

# Output (approx):
# Step 1:
# z = 0.5 → h ≈ 0.462
#
# Step 2:
# z = 1.1386 → h ≈ 0.813
#
# Step 3:
# z = 1.7439 → h ≈ 0.940
#
# Hidden States:
# [0.462, 0.813, 0.940]

# Gradient → small value showing how weights should change
gradient = 0.01
learning_rate = 0.1

# Update weights → adjust to improve future predictions
Wx = Wx - learning_rate * gradient
Wh = Wh - learning_rate * gradient

print("\nUpdated Wx:", Wx)
print("Updated Wh:", Wh)

# Output:
# Updated Wx: 0.499
# Updated Wh: 0.299