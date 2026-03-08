import math
import random

# Section 1: math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        # Avoid overflow for negative inputs
        z = math.exp(x)
        return z / (1 + z)

def d_sigmoid(y):
    # The derivative of sigmoid is y * (1 - y)
    # Note: 'y' here is the already activated value, not x.
    return y * (1.0 - y)


# Section 2: Neuron and Connection

class Connection:
    def __init__(self, source, target):
        self.source = source  # The Neuron sending the signal
        self.target = target  # The Neuron receiving the signal
        self.weight = random.uniform(-1, 1)
        self.gradient = 0.0   # Accumulates changes for the next update

    def assign_weight(self, w):
        # Helper to manually set weights for testing/debugging
        self.weight = w

class Neuron:
    def __init__(self, label=None):
        self.label = label          # For debugging (e.g., "Hidden 1")
        self.value = 0.0            # The current output (activation)
        self.bias = random.uniform(-1, 1)
        
        # Topology
        self.connections_in = []    # Dendrites (Incoming)
        self.connections_out = []   # Axons (Outgoing)
        
        # Backprop state
        self.delta = 0.0            # The local error signal
        self.bias_gradient = 0.0    # Accumulates changes for bias

    def forward(self):
        # 1. Sum up all (input_value * connection_weight)
        z = sum(conn.source.value * conn.weight for conn in self.connections_in)
        
        # 2. Add bias
        z += self.bias
        
        # 3. Activate
        self.value = sigmoid(z)
        return self.value

    def calculate_delta(self, target=None):
        # 1. Calculate the derivative of the activation function
        activation_derivative = d_sigmoid(self.value)
        
        if target is not None:
            # CASE A: Output Neuron
            # Error = (Output - Target)
            # Delta = Error * Derivative
            error = self.value - target
            self.delta = error * activation_derivative
        else:
            # CASE B: Hidden Neuron
            # We look at who we sent data to, and ask how much error they had.
            # Sum(outgoing_weight * outgoing_delta)
            sum_weighted_errors = sum(
                conn.weight * conn.target.delta 
                for conn in self.connections_out
            )
            self.delta = sum_weighted_errors * activation_derivative

    def update_weights(self, learning_rate):
        # Update the weights of connections COMING INTO this neuron
        # Weight_New = Weight_Old - (Learning_Rate * Input_Value * Local_Delta)
        for conn in self.connections_in:
            gradient = self.delta * conn.source.value
            conn.weight -= learning_rate * gradient

        # Update Bias
        # Bias_New = Bias_Old - (Learning_Rate * Local_Delta)
        self.bias -= learning_rate * self.delta


# Section 3: Network

class Network:
    def __init__(self):
        self.neurons = []
        self.input_neurons = []
        self.output_neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def set_inputs(self, neurons):
        self.input_neurons = neurons

    def set_outputs(self, neurons):
        self.output_neurons = neurons

    def connect(self, a, b):
        """Creates a synapse between Neuron A and Neuron B."""
        c = Connection(a, b)
        a.connections_out.append(c)
        b.connections_in.append(c)

    def predict(self, input_data):
        # 1. Set Input Values (Sensors)
        # We overwrite the values directly. These neurons do not 'fire' via math.
        for i, val in enumerate(input_data):
            self.input_neurons[i].value = val

        # 2. Forward Propagation
        # We loop through all neurons. If it's an input neuron, we skip it.
        # Note: This assumes self.neurons is added in a topological order 
        # (Input -> Hidden -> Output).
        for neuron in self.neurons:
            if neuron not in self.input_neurons:
                neuron.forward()

        # 3. Return Output
        return [n.value for n in self.output_neurons]

    def train(self, x, y, learning_rate):
        # 1. Forward Pass
        self.predict(x)

        # 2. Backward Pass (Calculate Gradients)
        # Step A: Calculate Delta for Output Neurons (Base Case)
        for i, neuron in enumerate(self.output_neurons):
            neuron.calculate_delta(target=y[i])

        # Step B: Calculate Delta for Hidden/Input Neurons (Recursive Step)
        # We iterate in REVERSE to ensure targets have calculated their delta 
        # before sources try to read them.
        for neuron in reversed(self.neurons):
            if neuron not in self.output_neurons:
                neuron.calculate_delta()

        # 3. Update Weights
        # Now that every neuron has its 'delta', we apply the changes.
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)