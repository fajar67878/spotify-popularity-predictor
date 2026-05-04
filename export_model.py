import pickle
import json
import numpy as np
import os

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print("model.pkl tidak ditemukan, training dulu...")
    from model_utils import train_model
    train_model()

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

nn         = bundle["nn"]
mean       = bundle["mean"]
std        = bundle["std"]
metrics    = bundle["metrics"]
history    = bundle["history"]
genre_cols = bundle["genre_cols"]

export = {
    "weights":     [w.tolist() for w in nn.weights],
    "biases":      [b.tolist() for b in nn.biases],
    "layer_sizes": nn.layer_sizes,
    "mean":        mean.tolist(),
    "std":         std.tolist(),
    "metrics":     metrics,
    "history":     history,
    "genre_cols":  genre_cols,
}

with open("model_weights.json", "w") as f:
    json.dump(export, f)

print(f"✅ Exported ke model_weights.json")