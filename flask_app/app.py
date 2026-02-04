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
from src.mlflow_config import setup_mlflow

# Setup MLflow
setup_mlflow()

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ============================================================
# Initialize NLTK data properly
# ============================================================
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("📥 Downloading WordNet...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("📥 Downloading OMW...")
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📥 Downloading stopwords...")
    nltk.download('stopwords', quiet=True)

# Pre-initialize to avoid lazy loading issues in multi-threaded Flask
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
print("✅ NLTK data initialized")
# ============================================================

def lemmatization(text):
    """Lemmatize the text."""
    text = text.split()
    text = [_lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    text = [word for word in str(text).split() if word not in _stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# Initialize Flask app
app = Flask(__name__)

# Create a custom registry for Prometheus
registry = CollectorRegistry()

# Define custom metrics
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", 
    ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", 
    ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", 
    ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "my_model"

def get_latest_model_version(model_name):
    """Get the latest model version from MLflow"""
    try:
        client = mlflow.MlflowClient()
        
        # Get ALL versions and sort by version number to get the absolute latest
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if all_versions:
            # Sort by version number (descending) and get the latest
            latest = sorted(all_versions, key=lambda x: int(x.version), reverse=True)[0]
            print(f"✅ Using latest model version: {latest.version} (stage: {latest.current_stage})")
            return latest.version
        
        # Fallback: try by stages
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        
        return latest_version[0].version if latest_version else None
    except Exception as e:
        print(f"❌ Error getting model version: {e}")
        return None

def load_model_and_vectorizer():
    """Load model from MLflow and vectorizer from local file"""
    try:
        # Get latest model version
        model_version = get_latest_model_version(model_name)
        
        if model_version:
            model_uri = f'models:/{model_name}/{model_version}'
            print(f"📦 Fetching model from: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
        else:
            # Fallback to local model
            print("⚠️  No model found in MLflow registry, loading from local file...")
            model_path = 'models/model.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        # Load vectorizer
        vectorizer_path = 'models/vectorizer.pkl'
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("✅ Model and vectorizer loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Vectorizer type: {type(vectorizer).__name__}")
        
        return model, vectorizer
    
    except Exception as e:
        print(f"❌ Error loading model or vectorizer: {e}")
        import traceback
        traceback.print_exc()
        raise

# Load model and vectorizer at startup
print("\n" + "="*60)
print("🚀 Loading model and vectorizer...")
print("="*60)
model, vectorizer = load_model_and_vectorizer()
print("="*60 + "\n")

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        text = request.form["text"]
        print(f"\n📝 Received text: {text[:100]}...")
        
        # Clean text
        text = normalize_text(text)
        print(f"🧹 Normalized text: {text[:100]}...")
        
        # Convert to features
        features = vectorizer.transform([text])
        print(f"🔢 Feature shape: {features.shape}")
        
        # Predict directly with sparse matrix
        prediction = model.predict(features)[0]
        print(f"🎯 Prediction: {prediction}")
        
        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        
        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        
        return render_template("index.html", result=int(prediction))
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", result=None, error=str(e))

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__
    }, 200

if __name__ == "__main__":
    print("\n🚀 Starting Flask application...")
    print("📍 Server will be available at: http://localhost:5000")
    print("📊 Metrics available at: http://localhost:5000/metrics")
    print("❤️  Health check at: http://localhost:5000/health\n")
    app.run(debug=True, host="0.0.0.0", port=5000)