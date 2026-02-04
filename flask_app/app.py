import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request
import mlflow
import mlflow.sklearn
import pickle
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ============================================================
# SAFE MLflow / Dagshub Initialization (NO import-time side effects)
# ============================================================
def init_mlflow_if_available():
    """
    Initialize MLflow/Dagshub only if tokens are present.
    Safe for local, CI, Docker, production.
    """
    try:
        from src.mlflow_config import setup_mlflow
        setup_mlflow()
        print("✅ MLflow initialized successfully")
    except Exception as e:
        print(f"⚠️ MLflow not initialized: {e}")
        print("➡️ Continuing without MLflow (safe mode)")

# ============================================================
# Initialize NLTK data
# ============================================================
import nltk

def init_nltk():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

init_nltk()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
print("✅ NLTK initialized")

# ============================================================
# Text preprocessing helpers
# ============================================================
def lemmatization(text):
    return " ".join([_lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return " ".join([w for w in text.split() if w not in _stop_words])

def removing_numbers(text):
    return ''.join([c for c in text if not c.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return re.sub('\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)

# Prometheus metrics
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction count",
    ["prediction"],
    registry=registry
)

# ============================================================
# Model loading
# ============================================================
MODEL_NAME = "my_model"

def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            return latest.version
    except Exception:
        return None

def load_model_and_vectorizer():
    try:
        model_version = get_latest_model_version(MODEL_NAME)

        if model_version:
            model_uri = f"models:/{MODEL_NAME}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Loaded model from MLflow: {model_uri}")
        else:
            with open("models/model.pkl", "rb") as f:
                model = pickle.load(f)
            print("⚠️ Loaded local model.pkl")

        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        print(f"❌ Failed to load model/vectorizer: {e}")
        raise

# ============================================================
# App startup sequence
# ============================================================
print("\n🚀 Initializing MLflow (optional)")
init_mlflow_if_available()

print("\n🚀 Loading model and vectorizer")
model, vectorizer = load_model_and_vectorizer()

# ============================================================
# Routes
# ============================================================
@app.route("/")
def home():
    REQUEST_COUNT.labels("GET", "/").inc()
    start = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels("/").observe(time.time() - start)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels("POST", "/predict").inc()
    start = time.time()

    try:
        text = request.form["text"]
        cleaned = normalize_text(text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        PREDICTION_COUNT.labels(str(prediction)).inc()
        REQUEST_LATENCY.labels("/predict").observe(time.time() - start)

        return render_template("index.html", result=int(prediction))

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template("index.html", result=None, error=str(e))

@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }, 200

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Flask app starting")
    print("📍 http://localhost:5000")
    print("📊 /metrics")
    print("❤️  /health\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
