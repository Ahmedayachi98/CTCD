from flask import Flask, request, render_template
import mlflow.pyfunc
import pickle
import numpy as np

# --------------------------------------------
# 1. MLflow Model Configuration
# --------------------------------------------
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Update with your MLflow URI if needed
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "SentimentModel"  # The registered model name in MLflow
STAGE = "production"

# Load the model dynamically from the "production" stage
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    print(f"Model in '{STAGE}' stage loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --------------------------------------------
# 2. Flask App Configuration
# --------------------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    """
    Render the homepage with a form for user input.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    if not model:
        return render_template('index.html', prediction_text="Model not loaded. Please check the configuration.")

    try:
        # Get text input from user
        text = request.form['text']
        if not text.strip():
            return render_template('index.html', prediction_text="Please enter valid text.")

        # Load vectorizer from MLflow if stored along with the model (as an artifact)
        with open('C:/Users/DELL/Desktop/CTCD/mlruns/1/a51733574bc34645becc50b7465f2adc/artifacts/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        #vectorizer = mlflow.pyfunc.load_model(vectorizer_uri)
        
        # Transform the input text using the loaded vectorizer
        text_vectorized = vectorizer.transform([text])

        # Predict sentiment
        prediction = model.predict(text_vectorized)[0]

        # Convert numeric prediction to sentiment label
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(prediction, "Unknown")

        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5011)
