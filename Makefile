export PYTHONPATH := $(PWD):$(PYTHONPATH)

.PHONY: start-mlflow \
	stop-mlflow \ 
	run-requirements \
	run-model \
	test-bentoml-model \
	serve-bentoml-model \
	build-bentoml-model \
	list-bentoml-artifacts \
	containerize-bentoml-model \
	serve-bentoml-model-docker

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

test-bentoml-model:
	@echo "Loading model to BentoML..."
	@python3 serving/iris-model-bento.py
	@echo "Done!"

serve-bentoml-model:
	@echo "Serving model with BentoML..."
	@bentoml serve serving.service:IrisClassifier
	@echo "Done!"

build-bentoml-model:
	@echo "Building BentoML model..."
	cd serving && bentoml build
	@echo "Done!"

containerize-bentoml-model:
	@echo "Containerizing BentoML model..."
	@echo "Targetting $(img):$(ver)"
	cd serving && bentoml containerize $(img):$(ver)
	@echo "Done!"

list-bentoml-artifacts:
	@echo "Listing BentoML artifacts..."
	bentoml list && bentoml models list
	@echo "Done!"

serve-bentoml-model-docker:
	@echo "Serving BentoML model in Docker..."
	@echo "Targetting $(img):$(ver)"
	docker run -it --rm -p 3000:3000 $(img):$(ver)
	@echo "Done!"