import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request
import mlflow
import mlflow.sklearn
import pickle
import time
import re
import string
import warnings

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ============================================================
# SAFE MLflow / DagsHub Initialization
# ============================================================
def init_mlflow_if_available():
    """
    Initialize MLflow only if credentials exist.
    Safe for local / CI / prod.
    """
    try:
        from src.mlflow_config import setup_mlflow
        setup_mlflow()
        print("✅ MLflow initialized")
    except Exception as e:
        print(f"⚠️ MLflow not initialized: {e}")
        print("➡️ Running in local-safe mode")

# ============================================================
# NLTK INIT
# ============================================================
def init_nltk():
    for pkg in ["wordnet", "omw-1.4", "stopwords"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

init_nltk()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
print("✅ NLTK initialized")

# ============================================================
# TEXT PREPROCESSING
# ============================================================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = "".join(c for c in text if not c.isdigit())
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    words = [w for w in text.split() if w not in _stop_words]
    words = [_lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

# ============================================================
# PROMETHEUS METRICS
# ============================================================
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total HTTP requests",
    ["method", "endpoint"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    registry=registry,
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Predictions by class",
    ["prediction"],
    registry=registry,
)

# ============================================================
# MODEL LOADING (PRODUCTION ONLY)
# ============================================================
MODEL_NAME = "my_model"

def get_production_model_version(model_name: str):
    """
    Returns the model version with tag env=production
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    for mv in versions:
        if mv.tags.get("env") == "production":
            print(f"✅ Found PRODUCTION model version: {mv.version}")
            return mv.version

    raise RuntimeError("❌ No model found with tag env=production")

def load_model_and_vectorizer():
    try:
        version = get_production_model_version(MODEL_NAME)
        model_uri = f"models:/{MODEL_NAME}/{version}"

        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Loaded production model: {model_uri}")

        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise

# ============================================================
# STARTUP SEQUENCE
# ============================================================
#docker ke liye isko hata rahe
#print("\n🚀 Initializing MLflow (optional)")
#init_mlflow_if_available()

#print("\n🚀 Loading production model")
#model, vectorizer = load_model_and_vectorizer()
#locally chalne ke liye isko le aao and niche wale ko comment out kar do

print("\n🚀 Loading model from local artifacts")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Local model loaded successfully")


# ============================================================
# ROUTES
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
        "vectorizer_loaded": vectorizer is not None,
        "model_name": MODEL_NAME,
    }, 200

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Flask app running")
    print("📍 http://localhost:5000")
    print("📊 /metrics")
    print("❤️  /health\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
