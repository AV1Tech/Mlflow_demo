import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["target"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run() as run:
    # Parameters
    n_estimators = 100
    max_depth = 3
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Train a model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train.values.ravel())

    # Predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Infer the model signature
    signature = infer_signature(X_train, clf.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(clf, "random_forest_model", signature=signature)

    # Register the model
    model_uri = f"runs:/{run.info.run_id}/random_forest_model"
    mlflow.register_model(model_uri, "IrisRandomForestModel")

    # Print out the run ID
    print(f"Run ID: {run.info.run_id}")
    print(f"Model URI: {model_uri}")
