import mlflow
from mlflow.models import infer_signature

import logging
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score



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

# Train logistic regression model
def train_model(X_train, X_test, y_train, y_test):
    # Initialize/train model
    # lr = LogisticRegression(
    #     random_state=8000,
    #     max_iter=1000,
    #     solver='lbfgs'
    # )
    lr = RandomForestClassifier(random_state=8000)

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5)
    logger.info(f'Cross-validation scores: {cv_scores}')
    logger.info(f'Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
    

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

# Orchestrator
def orchestrate():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    train_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    orchestrate()