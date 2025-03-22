.PHONY: start-mlflow stop-mlflow run-requirements run-model

start-mlflow:
	# run MLflow in detached mode
	mlflow server --host 127.0.0.1 --port 8080 &
	@echo "MLflow server started in detached mode"

stop-mlflow:
	@pkill -f "mlflow.server"
	@echo "MLflow server stopped"

run-requirements:
	@pip3 install -r requirements.txt

run-model:
	@echo "Running model..."
	@python3 models/iris-model.py