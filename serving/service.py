import bentoml
import numpy as np
import pandas as pd
from bentoml.models import BentoModel

demo_image = bentoml.images.PythonImage(python_version="3.12") \
    .python_packages("mlflow", "scikit-learn", "bentoml", "pandas")

target_names = ['setosa', 'versicolor', 'virginica']
feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                'petal length (cm)', 'petal width (cm)']

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    bento_model = BentoModel("iris-bento-model:ixhe44yhu6b2gabl")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data: np.ndarray) -> list[str]:
        # Convert numpy array to pandas DataFrame with correct column names
        df = pd.DataFrame(input_data, columns=feature_names)
        preds = self.model.predict(df)
        return [target_names[i] for i in preds]