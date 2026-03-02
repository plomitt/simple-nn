import math

# Data
X = [[0,0], [0,1],[1,0], [1,1]]
Y = [[0], [1], [1], [0]]

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