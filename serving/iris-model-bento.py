import bentoml
import numpy as np
import pandas as pd

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("serving.log")
    ]
)
logger = logging.getLogger(__name__)

model_tag = "iris-bento-model:latest"

def test_model():
    logger.info(f'Loading model {model_tag}')
    iris_model = bentoml.mlflow.load_model(model_tag)
    
    # Create DataFrame with correct column names
    input_data = pd.DataFrame(
        [[5.9, 3.0, 5.1, 1.8]], 
        columns=['sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'petal width (cm)']
    )
    logger.info(f'Testing for {input_data}')
    res = iris_model.predict(input_data)
    logger.info(f'Model species: {res}')

if __name__ == "__main__":
    test_model()

