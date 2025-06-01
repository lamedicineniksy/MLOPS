# Import libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
from google.cloud import aiplatform
import subprocess

# Constants
PROJECT_ID = "upbeat-palace-460620-p9"
LOCATION = "us-central1"
BUCKET_URI = "gs://mlops-course-upbeat-palace-460620-p9"
MODEL_ARTIFACT_DIR = "my-models/iris-classifier-week-1"

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

# Create the Cloud Storage bucket if it doesn't exist
def create_bucket(bucket_uri):
    try:
        # List existing buckets to check if bucket exists
        result = subprocess.run(["gsutil", "ls"], capture_output=True, text=True)
        if bucket_uri not in result.stdout:
            print(f"Creating bucket: {bucket_uri}")
            subprocess.run(["gsutil", "mb", "-l", LOCATION, "-p", PROJECT_ID, bucket_uri], check=True)
        else:
            print(f"Bucket {bucket_uri} already exists")
    except Exception as e:
        print(f"Error checking/creating bucket: {e}")

create_bucket(BUCKET_URI)

# Load iris data
data = pd.read_csv('iris.csv')

# Split data
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(metrics.accuracy_score(prediction, y_test)))

# Save model
os.makedirs('artifacts', exist_ok=True)
model_path = os.path.join('artifacts', 'model.joblib')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Upload model to GCS bucket
try:
    subprocess.run(["gsutil", "cp", model_path, f"{BUCKET_URI}/{MODEL_ARTIFACT_DIR}/"], check=True)
    print(f"Uploaded model.joblib to {BUCKET_URI}/{MODEL_ARTIFACT_DIR}/")
except Exception as e:
    print(f"Failed to upload model: {e}")
