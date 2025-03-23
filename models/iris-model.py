import mlflow
import bentoml
from mlflow.models import infer_signature
from datetime import date

import logging
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

mlflow_tracking_url = "http://127.0.0.1:8080"

# Set the logger
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)


# Load the Iris dataset
def load_dataset():
    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    logger.info(f'X shape: {X.shape}')
    logger.info(f'y shape: {y.shape}')
    return X, y

# Split dataset into train, test sets
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=True)
    logger.info(X_train.head())
    logger.info(X_test.head())
    return X_train, X_test, y_train, y_test

# Train logistic regression model, log to MLflow
def train_model(X_train, X_test, y_train, y_test):
    
    # set tracking url
    mlflow.set_tracking_uri(mlflow_tracking_url)
    mlflow.set_experiment("iris-model")
    
    with mlflow.start_run():
        # Initialize/train model
        # lr = LogisticRegression(
        #     random_state=8000,
        #     max_iter=1000,
        #     solver='lbfgs'
        # )
        params = {
            'n_estimators': 100,
            'random_state': 8000
        }
        lr = RandomForestClassifier(**params)
        lr.fit(X_train, y_train)

        # Predict on test set
        y_pred = lr.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        logger.info(f'Accuracy: {accuracy}')
        logger.info(f'Precision: {precision}')
        logger.info(f'Recall: {recall}')
        logger.info(f'F1: {f1}')

        # log artifacts to mlflow
        mlflow.log_params(params=params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("presicion", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.set_tag("Model info", "RandomForestClassifier for iris dataset")

        signature = infer_signature(X_train, lr.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            signature=signature,
            artifact_path="iris-model",
            input_example=X_train.head(),
            registered_model_name="iris-demo-model",
        )
        model_uri = mlflow.get_artifact_uri("iris-model")
        logger.info(f"Model saved at: {model_uri}")
        return model_uri

# Register model to Bentoml registry
def register_model(model_uri):
    mlflow.set_tracking_uri(mlflow_tracking_url)

    bento_model = bentoml.mlflow.import_model(
        name='iris-bento-model',
        model_uri=model_uri)
    return bento_model

# Orchestrator
def orchestrate():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model_uri = train_model(X_train, X_test, y_train, y_test)
    register_model(model_uri)


if __name__ == "__main__":
    orchestrate()