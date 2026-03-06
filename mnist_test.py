import os
import gzip
import urllib.request
from abstract import serial, Dense, ReLU, train_model, argmax

# ==========================================
# 1. PURE PYTHON MNIST DOWNLOADER & PARSER
# ==========================================

def download_mnist(filename, data_dir="data"):
    """Downloads the file from a reliable Google Storage mirror into a subfolder."""
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    # Create the subfolder if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename} into '{data_dir}/'...")
        urllib.request.urlretrieve(base_url + filename, filepath)
    
    return filepath

def load_images(filename, data_dir="data"):
    """Reads raw MNIST image bytes and converts them to a normalized list of lists (0.0 to 1.0)."""
    filepath = download_mnist(filename, data_dir)
    with gzip.open(filepath, 'rb') as f:
        f.read(16) # Skip the 16-byte header
        buf = f.read()
        
        # Each image is 28x28 = 784 pixels. 
        # We chunk the bytes into lists of 784, and normalize 0-255 to 0.0-1.0
        num_images = len(buf) // 784
        return [[b / 255.0 for b in buf[i * 784 : (i + 1) * 784]] for i in range(num_images)]

def load_labels(filename, data_dir="data"):
    """Reads raw MNIST label bytes and converts them to one-hot encoded lists."""
    filepath = download_mnist(filename, data_dir)
    with gzip.open(filepath, 'rb') as f:
        f.read(8) # Skip the 8-byte header
        buf = f.read()
        
        # Convert each label (0-9) into a one-hot list[0.0, 0.0, ..., 1.0, ...]
        labels =[]
        for b in buf:
            one_hot = [0.0] * 10
            one_hot[b] = 1.0
            labels.append(one_hot)
        return labels

# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

print("Loading MNIST Dataset...")
# We load the training data
X_train_full = load_images("train-images-idx3-ubyte.gz")
Y_train_full = load_labels("train-labels-idx1-ubyte.gz")

# We load the testing data
X_test_full = load_images("t10k-images-idx3-ubyte.gz")
Y_test_full = load_labels("t10k-labels-idx1-ubyte.gz")

# Split into batches
BATCH_SIZE = 1000
X_train = X_train_full[:BATCH_SIZE]
Y_train = Y_train_full[:BATCH_SIZE]

print(f"Loaded Training Data: {len(X_train)} samples of {len(X_train[0])} features.")

# ==========================================
# 3. DEFINE ARCHITECTURE
# ==========================================

print("Initializing Functional Neural Network...")
# A standard Multi-Layer Perceptron (MLP) architecture
network_init, network_forward = serial(
    Dense(128),
    ReLU(),
    Dense(64),
    ReLU(),
    Dense(10) # 10 output classes (digits 0-9)
)

# Initialize the weights dynamically using the 784 input dimension
final_dim, initial_params = network_init(in_dim=784)

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================

epochs = 100
learning_rate = 0.5

print("\n--- Starting Training ---")
trained_params, final_train_loss, final_train_acc = train_model(
    network_forward=network_forward,
    params=initial_params,
    X=X_train,
    Y=Y_train,
    epochs=epochs,
    learning_rate=learning_rate,
    print_every=10
)

# ==========================================
# 5. EVALUATE ON UNSEEN TEST DATA
# ==========================================

print("\n--- Evaluating on Test Set ---")
# To keep it fast in pure Python, we evaluate on the first 1,000 test images
X_test = X_test_full[:1000]
Y_test = Y_test_full[:1000]

# Run a single forward pass with the freshly trained parameters
test_logits, _ = network_forward(trained_params, X_test)

# Calculate Accuracy
test_preds = argmax(test_logits)
test_targets = argmax(Y_test)
correct = sum(1 for p, t in zip(test_preds, test_targets) if p == t)
test_accuracy = (correct / len(test_preds)) * 100

print(f"Test Accuracy: {test_accuracy:.1f}%")