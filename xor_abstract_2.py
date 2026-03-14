# --- Define the Architecture ---
from abstract_2 import Network, Neuron


net = Network()

# Create Neurons
bias = Neuron("Bias") # Optional: A bias neuron that always outputs 1
i1 = Neuron("Input 1")
i2 = Neuron("Input 2")
h1 = Neuron("Hidden 1")
h2 = Neuron("Hidden 2")
o1 = Neuron("Output 1")

# Register them in order (Order matters for calculation flow!)
net.add_neuron(i1)
net.add_neuron(i2)
net.add_neuron(h1)
net.add_neuron(h2)
net.add_neuron(o1)

# Define Roles
net.set_inputs([i1, i2])
net.set_outputs([o1])

# Wire them up (The Topology)
# Fully connected: Inputs -> Hiddens
net.connect(i1, h1)
net.connect(i1, h2)
net.connect(i2, h1)
net.connect(i2, h2)

# Hiddens -> Output
net.connect(h1, o1)
net.connect(h2, o1)

# --- Data (XOR) ---
data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# --- Training Loop ---
print("Training...")
for epoch in range(10000):
    total_error = 0
    for x, y in data:
        net.train(x, y, learning_rate=0.5)
        # Simple MSE tracking
        output = net.output_neurons[0].value
        total_error += (y[0] - output) ** 2
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Error {total_error:.4f}")

# --- Test ---
print("\nTesting:")
for x, y in data:
    prediction = net.predict(x)[0]
    print(f"In: {x} -> Out: {prediction:.4f} (Target: {y[0]})")