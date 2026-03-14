from abstract import serial, Dense, ReLU, train_model, argmax

# ==========================================
# 1. THE XOR DATASET
# ==========================================
print("Setting up XOR data...")

# Inputs (4 samples, 2 features)
X =[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

# Targets: One-hot encoded for 2 classes
# Class 0: [1.0, 0.0]
# Class 1: [0.0, 1.0]

Y = [
    [1.0, 0.0], # 0 XOR 0 = 0
    [0.0, 1.0], # 0 XOR 1 = 1
    [0.0, 1.0], # 1 XOR 0 = 1
    [1.0, 0.0]  # 1 XOR 1 = 0
]

# ==========================================
# 2. DEFINE ARCHITECTURE
# ==========================================
print("Initializing Functional Neural Network...")

# For XOR, a tiny network is enough. 
# We use 8 hidden neurons with ReLU, mapping to 2 output classes.
network_init, network_forward = serial(
    Dense(8),
    ReLU(),
    Dense(2) 
)

# Initialize the weights dynamically using the 2 input dimension
final_dim, initial_params = network_init(in_dim=2)

# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
epochs = 1000
learning_rate = 0.5 

print("\n--- Starting Training ---")
trained_params, final_train_loss, final_train_acc = train_model(
    network_forward=network_forward,
    params=initial_params,
    X=X,
    Y=Y,
    epochs=epochs,
    learning_rate=learning_rate,
    print_every=100
)

# ==========================================
# 4. SHOW FINAL PREDICTIONS
# ==========================================
print("\n--- Final Predictions ---")

# Run the inputs through the trained network one last time
final_logits, _ = network_forward(trained_params, X)
predictions = argmax(final_logits)
actuals = argmax(Y)

for i in range(len(X)):
    print(f"Input: {X[i]} | Predicted: {predictions[i]} | Actual: {actuals[i]}")
    
if final_train_acc == 100.0:
    print("\nSuccess! The network achieved a perfect score.")
else:
    print("\nNetwork got stuck in a local minimum. Run it again!")