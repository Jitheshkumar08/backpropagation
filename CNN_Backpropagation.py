import numpy as np

# Input image → a small 3x3 matrix representing pixel values
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Input Image:")
print(image)

# Output:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Kernel → a small filter used to detect patterns like edges
kernel = np.array([
    [1, 0],
    [0, -1]
])

print("\nInitial Kernel:")
print(kernel)

# Output:
# [[ 1  0]
#  [ 0 -1]]

# Convolution → sliding the kernel over the image step by step
output = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        # Select a 2x2 region from the image
        region = image[i:i+2, j:j+2]
        
        # Multiply region with kernel and sum all values
        output[i, j] = np.sum(region * kernel)

print("\nFeature Map:")
print(output)

# Output:
# [[-4. -4.]
#  [-4. -4.]]


# Gradient → small value showing how much kernel should change
gradient = np.ones_like(kernel) * 0.01

# Update kernel → adjust values to reduce error in future
lr = 0.1
kernel = kernel - lr * gradient

print("\nUpdated Kernel:")
print(kernel)

# Output:
# [[ 0.999 -0.001]
#  [-0.001 -1.001]]