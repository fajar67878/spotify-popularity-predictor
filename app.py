from flask import Flask, render_template, request, jsonify
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import ensure_model, predict_song, ALL_GENRES

app = Flask(__name__)


@app.route("/")
def index():
    bundle  = ensure_model()
    metrics = bundle["metrics"]

    # Hitung prediksi semua genre untuk grafik
    genre_predictions = {g: predict_song(g)["probability"] for g in ALL_GENRES}

    return render_template("index.html",
                           metrics=metrics,
                           genres=ALL_GENRES,
                           genre_predictions=json.dumps(genre_predictions))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json()
        genre  = data.get("genre", "pop")
        result = predict_song(genre)
        if result is None:
            return jsonify({"error": "Model gagal dimuat!"}), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
