import numpy as np
import pandas as pd
import pickle
import json
import os
from neural_network import NeuralNetwork

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.json")

ALL_GENRES = [
    'pop', 'rock', 'jazz', 'classical', 'hip-hop', 'afrobeats', 'latin',
    'indian', 'country', 'r&b', 'electronic', 'soul', 'gaming', 'j-pop',
    'metal', 'reggae', 'k-pop', 'arabic', 'punk', 'blues', 'folk', 'lofi',
    'brazilian', 'turkish', 'ambient', 'korean', 'world', 'indie', 'mandopop',
    'cantopop', 'wellness', 'gospel', 'funk', 'soca', 'disco'
]

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

# Cache model di memory agar tidak load ulang tiap request
_cached_bundle = None


def load_model():
    global _cached_bundle
    if _cached_bundle is not None:
        return _cached_bundle

    # Coba pkl dulu (lokal)
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _cached_bundle = pickle.load(f)
                return _cached_bundle
        except:
            pass

    # Fallback ke JSON (Vercel)
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "r") as f:
            data = json.load(f)
        nn = NeuralNetwork(layer_sizes=data["layer_sizes"], learning_rate=0.005)
        nn.weights = [np.array(w) for w in data["weights"]]
        nn.biases  = [np.array(b) for b in data["biases"]]
        _cached_bundle = {
            "nn":         nn,
            "mean":       np.array(data["mean"]),
            "std":        np.array(data["std"]),
            "metrics":    data["metrics"],
            "history":    data["history"],
            "genre_cols": data["genre_cols"],
        }
        return _cached_bundle

    return None


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def ensure_model():
    """Load model dari JSON — tidak perlu training di Vercel."""
    bundle = load_model()
    if bundle is None:
        raise RuntimeError("model_weights.json tidak ditemukan! Jalankan export_model.py dulu.")
    return bundle


def predict_song(genre: str):
    bundle     = ensure_model()
    nn         = bundle["nn"]
    mean       = bundle["mean"]
    std        = bundle["std"]
    genre_cols = bundle["genre_cols"]

    X = np.zeros((1, len(genre_cols)))
    target_col = f"genre_{genre}"
    if target_col in genre_cols:
        X[0, genre_cols.index(target_col)] = 1

    X_n, _, _ = normalize(X, mean, std)
    prob  = float(nn.predict(X_n)[0][0])
    label = "🔥 High Popularity" if prob >= 0.5 else "📉 Low Popularity"
    return {
        "probability": round(prob * 100, 1),
        "label": label,
        "confidence": round(abs(prob - 0.5) * 200, 1)
    }
