import math
import random

# ==========================================
# SECTION 1: CORE MATRIX OPERATIONS
# ==========================================

def transpose(matrix):
    """Swaps rows and columns."""
    return[list(row) for row in zip(*matrix)]

def dot_product(matrix_a, matrix_b):
    """Standard matrix multiplication (A @ B)."""
    b_transposed = transpose(matrix_b)
    return[[sum(a * b for a, b in zip(row_a, col_b)) for col_b in b_transposed]
        for row_a in matrix_a
    ]

def add_matrices(A, B):
    """Adds two matrices. Supports broadcasting if B is a 1-row bias matrix."""
    if len(B) == 1:
        return [[a + b for a, b in zip(row_a, B[0])] for row_a in A]
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

def subtract_matrices(A, B):
    """Element-wise subtraction."""
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]

def multiply_scalar(matrix, scalar):
    """Multiplies every element by a scalar."""
    return [[val * scalar for val in row] for row in matrix]

def sum_columns(matrix):
    """Sums matrix along columns, collapsing it to a 1-row matrix (used for bias gradients)."""
    return [[sum(col) for col in zip(*matrix)]]

def initialize_matrix(rows, cols):
    """
    Creates a rows x cols matrix.
    Uses Xavier/Glorot initialization scaling to prevent exploding/vanishing 
    gradients in larger networks like MNIST.
    """
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

# ==========================================
# 2. ACTIVATIONS & DERIVATIVES
# ==========================================

def relu(matrix):
    """Rectified Linear Unit: max(0, x). Great for hidden layers."""
    return [[val if val > 0 else 0.0 for val in row] for row in matrix]

def relu_derivative(matrix):
    """Derivative of ReLU: 1 if x > 0 else 0."""
    return [[1.0 if val > 0 else 0.0 for val in row] for row in matrix]

def softmax(matrix):
    """
    Converts a row of numbers into a probability distribution.
    Used for the final layer of multi-class classification (MNIST).
    """
    result =[]
    for row in matrix:
        # Subtract max for numerical stability (prevents math.exp overflow)
        m = max(row)
        exps = [math.exp(val - m) for val in row]
        sum_exps = sum(exps)
        result.append([e / sum_exps for e in exps])
    return result

# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================

def cross_entropy_loss(predictions, targets):
    """
    Calculates the loss for classification. 
    Assumes `targets` is one-hot encoded (e.g.,[0, 0, 1, 0...]).
    """
    epsilon = 1e-15 # Prevents log(0) error
    loss = 0.0
    for p_row, t_row in zip(predictions, targets):
        for p, t in zip(p_row, t_row):
            if t == 1.0: 
                loss -= math.log(max(p, epsilon))
    return loss / len(predictions)

def cross_entropy_softmax_gradient(predictions, targets):
    """
    The mathematical miracle of Deep Learning: 
    The combined gradient of Softmax AND Cross Entropy is simply (Predictions - Targets).
    """
    return subtract_matrices(predictions, targets)

# ==========================================
# 4. UTILITIES
# ==========================================

def argmax(matrix):
    """Returns the index of the maximum value in each row (used to calculate accuracy)."""
    return [row.index(max(row)) for row in matrix]

# ==========================================
# SECTION 2: VECTOR-JACOBIAN PRODUCTS (VJP)
# ==========================================

def dot_product_vjp(A, B):
    """
    Forward: Calculates A @ B.
    Backward: Returns the gradient w.r.t A, and the gradient w.r.t B.
    """
    out = dot_product(A, B)
    
    def backward(grad_out):
        # The chain rule for matrix multiplication
        grad_A = dot_product(grad_out, transpose(B))
        grad_B = dot_product(transpose(A), grad_out)
        return grad_A, grad_B
        
    return out, backward

def add_matrices_vjp(A, B):
    """
    Forward: Calculates A + B.
    Backward: Returns the gradient w.r.t A, and the gradient w.r.t B.
    """
    out = add_matrices(A, B)
    
    def backward(grad_out):
        # The derivative of addition is exactly 1. 
        # So grad_A gets the raw gradient passed straight through.
        grad_A = grad_out 
        
        # Because B is broadcasted across the batch (1-row bias), 
        # its gradient is the sum of the gradients across the batch (columns).
        grad_B = sum_columns(grad_out) 
        
        return grad_A, grad_B
        
    return out, backward

def relu_vjp(Z):
    """
    Forward: Calculates ReLU(Z).
    Backward: Applies the ReLU derivative to the incoming gradient.
    """
    out = relu(Z)
    
    def backward(grad_out):
        # Element-wise multiplication: incoming gradient * ReLU derivative
        # We can write this inline beautifully using our raw lists
        grad_Z = [[g * (1.0 if z > 0 else 0.0) for g, z in zip(row_g, row_z)]
            for row_g, row_z in zip(grad_out, Z)
        ]
        return grad_Z
        
    return out, backward

def cross_entropy_softmax_vjp(logits, targets):
    """
    Forward: Computes softmax probabilities, then the scalar cross-entropy loss.
    Backward: Computes the miraculously simple combined gradient w.r.t logits.
    """
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, targets)
    
    def backward(grad_out=1.0):
        # The combined derivative of Softmax + CrossEntropy is (probs - targets).
        # We average the gradient over the batch size so the learning rate 
        # remains stable regardless of how much data we pass in.
        batch_size = len(logits)
        grad_logits = [[(p - t) * grad_out / batch_size for p, t in zip(p_row, t_row)]
            for p_row, t_row in zip(probs, targets)
        ]
        return grad_logits
        
    return loss, backward

# ==========================================
# SECTION 3: FUNCTIONAL LAYERS & COMBINATORS
# ==========================================

def Dense(out_dim):
    """A fully connected neural network layer."""
    
    def init_fn(in_dim):
        # Dynamically create weights based on whatever the input dimension is!
        params = {
            "w": initialize_matrix(in_dim, out_dim),
            "b": initialize_matrix(1, out_dim)
        }
        return out_dim, params
        
    def apply_fn(params, X):
        # 1. Forward Pass (chaining our VJPs)
        Z, vjp_dot = dot_product_vjp(X, params["w"])
        out, vjp_add = add_matrices_vjp(Z, params["b"])
        
        # 2. Backward Closure
        def backward(grad_out):
            # Chain rule in reverse
            grad_Z, grad_b = vjp_add(grad_out)
            grad_X, grad_w = vjp_dot(grad_Z)
            
            # Return the gradient of the input (to pass further back) 
            # AND the gradients for this layer's parameters
            return grad_X, {"w": grad_w, "b": grad_b}
            
        return out, backward

    return init_fn, apply_fn

def ReLU():
    """An activation layer."""
    
    def init_fn(in_dim):
        # Activations don't change the dimension, and have no parameters (weights/biases)
        return in_dim, {}
        
    def apply_fn(params, X):
        out, vjp_relu = relu_vjp(X)
        
        def backward(grad_out):
            grad_X = vjp_relu(grad_out)
            # Return input gradient, and an empty dict for parameter gradients
            return grad_X, {} 
            
        return out, backward

    return init_fn, apply_fn

def serial(*layers):
    """
    The magic combinator. Takes a sequence of layers and returns a 
    single unified (init_fn, apply_fn) that routes everything automatically.
    """
    
    def init_fn(in_dim):
        all_params =[]
        current_dim = in_dim
        
        for layer_init, _ in layers:
            # Pass the dimension through the network to size matrices automatically
            current_dim, layer_params = layer_init(current_dim)
            all_params.append(layer_params)
            
        # Returns the final output dimension, and a list of dictionaries containing all weights
        return current_dim, all_params
        
    def apply_fn(all_params, X):
        vjp_functions =[]
        current_X = X
        
        # 1. Master Forward Pass: Loop through all layers automatically
        for layer_params, (_, layer_apply) in zip(all_params, layers):
            current_X, vjp = layer_apply(layer_params, current_X)
            vjp_functions.append(vjp)
            
        # 2. Master Backward Closure: Execute all closures in reverse!
        def backward(grad_out):
            all_param_grads =[]
            current_grad = grad_out
            
            # Iterate backwards through our saved closures
            for vjp in reversed(vjp_functions):
                current_grad, param_grads = vjp(current_grad)
                all_param_grads.append(param_grads)
                
            # Reverse the collected gradients so they match the order of all_params
            return current_grad, all_param_grads[::-1]
            
        return current_X, backward

    return init_fn, apply_fn

# ==========================================
# SECTION 4: FUNCTIONAL OPTIMIZER (SGD)
# ==========================================

def sgd_update(params, grads, learning_rate):
    """
    Pure functional Stochastic Gradient Descent.
    Takes a list of layer parameters and layer gradients, 
    and returns a beautifully fresh, updated list of parameters.
    """
    new_params =[]
    
    # zip pairs up the parameters and gradients for each layer sequentially
    for layer_params, layer_grads in zip(params, grads):
        new_layer_params = {}
        
        # Keys will be "w" and "b" (or empty if it's a ReLU layer)
        for key in layer_params:
            # step = grad * learning_rate
            step = multiply_scalar(layer_grads[key], learning_rate)
            # new_param = old_param - step
            new_layer_params[key] = subtract_matrices(layer_params[key], step)
            
        new_params.append(new_layer_params)
        
    return new_params


# ==========================================
# SECTION 5: THE SOTA FUNCTIONAL TRAINING LOOP
# ==========================================

def train_model(network_forward, params, X, Y, epochs, learning_rate, print_every=10):
    """
    Pure functional training loop.
    Maps (initial_params, data) -> (trained_params, final_loss, final_accuracy).
    """
    # We use a local reference so we don't mutate the original params passed in
    current_params = params
    
    for epoch in range(epochs + 1):
        
        # 1. Forward Pass
        logits, master_backward = network_forward(current_params, X)
        
        # 2. Compute Loss & Gradients
        loss, loss_backward = cross_entropy_softmax_vjp(logits, Y)
        grad_logits = loss_backward() 
        _, grads = master_backward(grad_logits)
        
        # 3. Purely update parameters (returns a brand new parameter dictionary)
        current_params = sgd_update(current_params, grads, learning_rate)
        
        # 4. Logging
        if epoch % print_every == 0:
            preds = argmax(logits)
            targets = argmax(Y)
            correct = sum(1 for p, t in zip(preds, targets) if p == t)
            accuracy = (correct / len(preds)) * 100
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {accuracy:.1f}%")
            
    # Calculate final metrics with the fully trained parameters
    final_logits, _ = network_forward(current_params, X)
    final_loss, _ = cross_entropy_softmax_vjp(final_logits, Y)
    
    final_preds = argmax(final_logits)
    final_targets = argmax(Y)
    final_correct = sum(1 for p, t in zip(final_preds, final_targets) if p == t)
    final_accuracy = (final_correct / len(final_preds)) * 100
    
    return current_params, final_loss, final_accuracy

# ==========================================
# USAGE
# ==========================================

if __name__ == "__main__":
    # 1. Define the Architecture
    network_init, network_forward = serial(
        Dense(128),
        ReLU(),
        Dense(64),
        ReLU(),
        Dense(10)
    )

    # 2. Initialize the starting parameters
    final_dim, initial_params = network_init(in_dim=784)

    # 3. Load your data (Mocking a batch of 32 images here)
    X_batch = initialize_matrix(32, 784) 
    Y_batch = [[1.0 if i == 3 else 0.0 for i in range(10)] for _ in range(32)] 

    # 4. Run the training function!
    print("Starting Training...")
    trained_params, final_loss, final_acc = train_model(
        network_forward=network_forward,
        params=initial_params,
        X=X_batch,
        Y=Y_batch,
        epochs=10,
        learning_rate=0.01,
        print_every=20
    )

    print("-" * 30)
    print(f"Training Complete!")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Accuracy: {final_acc:.1f}%")