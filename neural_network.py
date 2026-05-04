import numpy as np


class NeuralNetwork:
    """
    Backpropagation Neural Network dari scratch.
    Arsitektur: Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
    """

    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        layer_sizes: list jumlah neuron tiap layer, contoh [10, 16, 8, 1]
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        self.training_history = []

        # Inisialisasi bobot dengan He initialization
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    # ── Activation functions ──────────────────────────────────────────────────

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Meneruskan input X melalui semua layer.
        Mengembalikan list aktivasi tiap layer.
        """
        activations = [X]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            pre_activations.append(z)

            # Hidden layers pakai ReLU, output layer pakai Sigmoid
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)

            activations.append(a)

        return activations, pre_activations

    # ── Loss ─────────────────────────────────────────────────────────────────

    def binary_cross_entropy(self, y_pred, y_true):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ── Backward pass (Backpropagation) ──────────────────────────────────────

    def backward(self, activations, pre_activations, y_true):
        """
        Menghitung gradient menggunakan chain rule (backpropagation).
        Update bobot dan bias.
        """
        m = y_true.shape[0]
        n_layers = len(self.weights)

        grad_W = [None] * n_layers
        grad_b = [None] * n_layers

        # Delta output layer (sigmoid + binary cross-entropy)
        delta = activations[-1] - y_true  # dL/dz_output

        for i in reversed(range(n_layers)):
            grad_W[i] = (activations[i].T @ delta) / m
            grad_b[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                # Propagasi balik ke layer sebelumnya
                delta = (delta @ self.weights[i].T) * self.relu_derivative(pre_activations[i - 1])

        # Update bobot dengan gradient descent
        for i in range(n_layers):
            self.weights[i] -= self.lr * grad_W[i]
            self.biases[i] -= self.lr * grad_b[i]

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X, y, epochs=200, batch_size=64, verbose=True):
        """
        Melatih neural network dengan mini-batch gradient descent.
        """
        self.training_history = []
        m = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            idx = np.random.permutation(m)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            epoch_loss = 0
            n_batches = 0

            # Mini-batch
            for start in range(0, m, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                activations, pre_activations = self.forward(X_batch)
                loss = self.binary_cross_entropy(activations[-1], y_batch)
                self.backward(activations, pre_activations, y_batch)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            # Hitung akurasi pada full dataset tiap 10 epoch
            if epoch % 10 == 0 or epoch == epochs - 1:
                y_pred_full = self.predict(X)
                acc = np.mean((y_pred_full >= 0.5) == y.flatten()) * 100
                record = {"epoch": int(epoch + 1), "loss": round(float(avg_loss), 4), "accuracy": round(float(acc), 2)}
                self.training_history.append(record)
                if verbose:
                    print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        return self.training_history

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def predict_class(self, X):
        probs = self.predict(X)
        return (probs >= 0.5).astype(int).flatten()
