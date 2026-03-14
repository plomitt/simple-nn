import os
import gzip
import urllib.request
from abstract_3 import Network, Linear, ReLU, Sigmoid, MSE, Tanh

# ==========================================
# 1. PURE PYTHON MNIST DOWNLOADER & PARSER
# ==========================================

def download_mnist(filename, data_dir="data"):
    """Downloads the file from a reliable Google Storage mirror into a subfolder."""
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
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
        
        num_images = len(buf) // 784
        return [[b / 255.0 for b in buf[i * 784 : (i + 1) * 784]] for i in range(num_images)]

def load_labels(filename, data_dir="data"):
    """Reads raw MNIST label bytes and converts them to one-hot encoded lists."""
    filepath = download_mnist(filename, data_dir)
    with gzip.open(filepath, 'rb') as f:
        f.read(8) # Skip the 8-byte header
        buf = f.read()
        
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
X_train_full = load_images("train-images-idx3-ubyte.gz")
Y_train_full = load_labels("train-labels-idx1-ubyte.gz")

X_test_full = load_images("t10k-images-idx3-ubyte.gz")
Y_test_full = load_labels("t10k-labels-idx1-ubyte.gz")

# Split into batches
BATCH_SIZE = 10000
X_train = X_train_full[:BATCH_SIZE]
Y_train = Y_train_full[:BATCH_SIZE]

X_test = X_test_full[:BATCH_SIZE]
Y_test = Y_test_full[:BATCH_SIZE]

print(f"Loaded Training Data: {len(X_train)} samples of {len(X_train[0])} features.")

# ==========================================
# 3. DEFINE ARCHITECTURE
# ==========================================

print("Building CNN Topology...")
net = Network()

# Input Layer (28x28 = 784)
net.add_layer(784, Linear)

# Conv Layer (3x3 kernel). Output: 26x26 grid
w, h = net.add_conv_filter(img_width=28, img_height=28, kernel_size=3, activation=Tanh)

# Max Pool (2x2). Output: 13x13 grid
w, h = net.add_max_pool_layer(img_width=w, img_height=h, pool_size=2)

# Dense Hidden Layer (32 neurons to keep pure python loops faster)
net.add_layer(32, Tanh)

# Output Layer (10 digits)
net.add_layer(10, Sigmoid)

print(f"Network created with {len(net.layers)} layers!")

# Train using the separated MSE class
net.fit(X_train, Y_train, epochs=10, learning_rate=0.05, loss_cls=MSE)

print("\nTraining Complete! Evaluating on separate Test Set...")
accuracy = net.evaluate(X_test, Y_test, verbose=True)