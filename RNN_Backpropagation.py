import numpy as np

# Initialize weights → random values for input and hidden connections
Wx = np.random.randn()   # weight for input
Wh = np.random.randn()   # weight for hidden state

print("Initial Wx:", Wx)
print("Initial Wh:", Wh)

# Initial hidden state → starts with zero (no memory)
h_prev = 0

# Input sequence → sequential data (like time steps)
inputs = [1, 2, 3]

hidden_states = []

# Forward pass → process each input step by step
for x in inputs:
    # Combine current input and previous hidden state
    h = np.tanh(Wx * x + Wh * h_prev)
    
    # Store hidden state
    hidden_states.append(h)
    
    # Update previous hidden state (memory)
    h_prev = h

print("\nHidden States:")
print(hidden_states)

# Gradient → small value showing how weights should change
gradient = 0.01
learning_rate = 0.1

# Update weights → adjust to improve future predictions
Wx = Wx - learning_rate * gradient
Wh = Wh - learning_rate * gradient

print("\nUpdated Wx:", Wx)
print("Updated Wh:", Wh)