"""Small HTTP API for the department classifier; this service does not use MySQL."""

from flask import Flask, jsonify, request

from model.nlp_model import VALID_DEPARTMENTS, load_model, predict_department

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify({
        "service": "Campus Grievance Department Classifier",
        "health": "/health",
        "prediction": {"method": "POST", "path": "/predict"},
    })


@app.get("/health")
def health():
    if load_model() is None:
        return jsonify({"status": "model_unavailable"}), 503
    return jsonify({"status": "ready", "departments": VALID_DEPARTMENTS})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with a description field."}), 400

    description = payload.get("description", payload.get("text"))
    if not isinstance(description, str) or not description.strip():
        return jsonify({"error": "Description must be a non-empty string."}), 400
    if len(description) > 5000:
        return jsonify({"error": "Description must be 5000 characters or fewer."}), 400

    return jsonify({"department": predict_department(description.strip())})
