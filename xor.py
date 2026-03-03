import math
import random

# Math helpers
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def dot_product(matrix_a, matrix_b):
    b_transposed = transpose(matrix_b)
    
    return [
        [
            sum(a_val * b_val for a_val, b_val in zip(row_a, col_b))
            for col_b in b_transposed
        ]
        for row_a in matrix_a
    ]

def add_matrices(A, B):
    # Determine if B is a single-row matrix (bias broadcasting)
    is_bias = len(B) == 1
    
    if is_bias:
        return [[a + b for a, b in zip(row_a, B[0])] for row_a in A]
    else:
        return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

def subtract_matrices(A, B):
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

def hadamard_product(A, B):
    return [[a * b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

def multiply_scalar(matrix, scalar):
    return [[val * scalar for val in row] for row in matrix]

def substract_scalar(matrix, scalar):
    return [[scalar - val for val in row] for row in matrix]

def sigmoid(matrix):
    # Creating a local reference to math.exp is a micro-optimization 
    # that speeds up the inner loop significantly in native Python.
    exp = math.exp
    return [[1 / (1 + exp(-val)) for val in row] for row in matrix]

def sum_columns(matrix):
    return [[sum(col) for col in zip(*matrix)]]

def initialize_matrix(rows, cols):
    """Creates a rows x cols matrix with random values between -1 and 1."""
    runif = random.uniform
    return [[runif(-1, 1) for _ in range(cols)] for _ in range(rows)]

def mean_squared_error(matrix):
    sq_error = hadamard_product(matrix, matrix)
    sq_error_sum = sum_columns(sq_error)
    mean_sq_error = sq_error_sum[0][0] / len(matrix)
    return mean_sq_error


# 1. Data
X = [[0,0], [0,1], [1,0], [1,1]] # 4x2
Y = [[0], [1], [1], [0]]         # 4x1

# 2. Init network
epochs = 10000
learning_rate = 0.5
w1 = initialize_matrix(2, 2)     # 2x2
b1 = initialize_matrix(1, 2)     # 1x2
w2 = initialize_matrix(2, 1)     # 2x1
b2 = initialize_matrix(1, 1)     # 1x1

# 3. Training loop
for epoch in range(epochs+1):
    # Forward pass
    z1 = add_matrices(dot_product(X, w1), b1) # 4x2
    a1 = sigmoid(z1)                          # 4x2

    z2 = add_matrices(dot_product(a1, w2), b2) # 4x1
    a2 = sigmoid(z2)                           # 4x1

    # Backpropagation
    e2 = subtract_matrices(Y, a2) # 4x1
    d2 = hadamard_product(e2, hadamard_product(a2, substract_scalar(a2, 1))) # 4x1

    e1 = dot_product(d2, transpose(w2)) # 4x2
    d1 = hadamard_product(e1, hadamard_product(a1, substract_scalar(a1, 1))) # 4x2

    # Gradient descent
    w_u2 = multiply_scalar(dot_product(transpose(a1), d2), learning_rate) # 2x1
    w2 = add_matrices(w2, w_u2) # 2x1
    
    b_u2 = multiply_scalar(sum_columns(d2), learning_rate) # 1x1
    b2 = add_matrices(b2, b_u2) # 1x1

    w_u1 = multiply_scalar(dot_product(transpose(X), d1), learning_rate) # 2x2
    w1 = add_matrices(w1, w_u1) # 2x2
    
    b_u1 = multiply_scalar(sum_columns(d1), learning_rate) # 1x2
    b1 = add_matrices(b1, b_u1) # 1x2

    if epoch % 1000 == 0:
        loss = mean_squared_error(e2)
        print(f">>> Epoch {epoch} | loss {loss:.3f}")

# 4. Final forward pass
z1 = add_matrices(dot_product(X, w1), b1)
a1 = sigmoid(z1)
z2 = add_matrices(dot_product(a1, w2), b2)
a2 = sigmoid(z2)


# Visualisation helpers
def print_matrix(label, matrix):
    print()
    print(f"{label}:")
    for row in matrix:
        formatted_row = "  ".join([f"{val:7.3f}" for val in row])
        print(f"  [ {formatted_row} ]")

def show_performance(X, Y, predictions):
    print("\n--- SYSTEM INFERENCE REPORT ---")
    print("IN1  IN2  |  TARGET  |  PREDICT  |  STATUS")
    print("-" * 42)
    for i in range(len(X)):
        target = Y[i][0]
        pred = predictions[i][0]
        
        status = "[  OK  ]" if round(pred) == target else "[ FAIL ]"
        
        ix1, ix2 = X[i]
        print(f" {ix1}    {ix2}   |    {target}     |  {pred:7.4f}  | {status}")
    print("-" * 42 + "\n")

def draw_brain(W1, B1, W2, B2):
    # Mapping for clarity
    w1_11, w1_12 = W1[0][0], W1[0][1]
    w1_21, w1_22 = W1[1][0], W1[1][1]
    b1_1,  b1_2  = B1[0][0], B1[0][1]
    w2_1,  w2_2  = W2[0][0], W2[1][0]
    b_out        = B2[0][0]

    print("\n--- NETWORK ARCHITECTURE ---")
    print(f"IN 1 ----({w1_11:6.2f})----> HID 1 (bias:{b1_1:5.2f})")
    print(f"      \\               /          \\")
    print(f"       --({w1_12:5.2f})--\\ /            --({w2_1:5.2f})--+")
    print(f"                   X                        |---> OUT (bias:{b_out:5.2f})")
    print(f"       --({w1_21:5.2f})--/ \\            --({w2_2:5.2f})--+")
    print(f"      /               \\          /")
    print(f"IN 2 ----({w1_22:6.2f})----> HID 2 (bias:{b1_2:5.2f})")
    print("-" * 60)


# 5. Visualize results
print_matrix("Final Hidden Weights (W1)", w1)
print_matrix("Final Output Weights (W2)", w2)
draw_brain(w1, b1, w2, b2)
show_performance(X, Y, a2)