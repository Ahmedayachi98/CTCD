# Import libraries
import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import pickle
 
# Set up FastAPI
app = FastAPI()
 
# --------------------------------------------
# 1. MLflow and Dataset Configuration
# --------------------------------------------
# Set MLflow tracking URI and experiment
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment-Analysis-Experiment")
 
# Path to the dataset
DATASET_PATH = "./Twitter_Data.csv"
 
# --------------------------------------------
# 2. Training Function
# --------------------------------------------
def train_model(dataset):
    # Prepare dataset
    print("Training model...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(dataset['clean_text'])
    y = dataset['category'].map({'negative': 0, 'neutral': 1, 'positive': 2})
 
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
 
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
 
    # Log model and metrics to MLflow
    with mlflow.start_run():
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        signature = infer_signature(X_train.toarray(), y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sentiment_model",
            signature=signature,
            input_example=X_train[0].toarray(),
            registered_model_name="SentimentModel"
        )
        # Log the vectorizer as an artifact
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact('vectorizer.pkl')  # Log the vectorizer file
        
    print(f"Model training complete. Accuracy: {accuracy}, F1 Score: {f1}")
    return vectorizer, model
 
# Load and train the initial model
twitter = pd.read_csv(DATASET_PATH)
twitter.dropna(subset=['clean_text', 'category'], inplace=True)
twitter['category'] = twitter['category'].replace({-1: 'negative', 0: 'neutral', 1: 'positive'})
vectorizer, model = train_model(twitter)
 
# --------------------------------------------
# 3. FastAPI Deployment
# --------------------------------------------
# Input data schema
class SentimentInput(BaseModel):
    text: List[str]  # List of sentences for sentiment analysis
 
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is up and running!"}
 
@app.post("/predict/")
def predict_sentiment(input_data: SentimentInput):
    try:
        text_vectorized = vectorizer.transform(input_data.text)
        predictions = model.predict(text_vectorized)
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiments = [sentiment_map[pred] for pred in predictions]
        return {"sentiments": sentiments}
    except Exception as e:
        return {"error": str(e)}
 
# --------------------------------------------
# 4. File Monitoring for Retraining
# --------------------------------------------
class CSVFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.csv'):
            return
        print(f"Detected change in: {event.src_path}")
        try:
            new_dataset = pd.read_csv(event.src_path)
            new_dataset.dropna(subset=['clean_text', 'category'], inplace=True)
            new_dataset['category'] = new_dataset['category'].replace({-1: 'negative', 0: 'neutral', 1: 'positive'})
            global vectorizer, model
            vectorizer, model = train_model(new_dataset)
            print("Model retrained successfully.")
        except Exception as e:
            print(f"Error during retraining: {e}")
 
# Start file monitoring
print("Starting file monitoring...")
os.makedirs("./Datasets", exist_ok=True)
path_to_watch = "./Datasets"
event_handler = CSVFileHandler()
observer = Observer()
observer.schedule(event_handler, path=path_to_watch, recursive=False)
observer.start()
 
# --------------------------------------------
# 5. Run FastAPI Server and Observer
# --------------------------------------------
if __name__ == "__main__":
    try:
        print("Running API server...")
        uvicorn.run(app, host="127.0.0.1", port=8001)
    except KeyboardInterrupt:
        print("Stopping observer and server...")
        observer.stop()
        observer.join()
import matplotlib.pyplot as plt
import seaborn as sns
 
# Visualize sentiment distribution in the dataset
sentiment_counts = twitter['category'].value_counts()
 
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title('Sentiment Distribution in Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()