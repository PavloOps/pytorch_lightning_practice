.PHONY: help install train-smoke train visualize-network predict lint

PYTHON ?= python
DATA_DIR ?= ./data/raw
WEIGHTS_PATH ?= ./models/convnext_food101.ckpt
ONNX_PATH ?= ./models/convnext_food101.onnx
PREDICTIONS_PATH ?= ./data/predictions/results.csv
IMAGE_DIR ?= ./data/raw/food-101/images
LR ?= 0.0003

help:
	@echo "Available commands:"
	@echo "  make install      Install project requirements"
	@echo "  make train-smoke  Run fast development training check"
	@echo "  make train        Run training"
	@echo "  make visualize-network  Save model graph locally"
	@echo "  make predict      Run prediction"
	@echo "  make lint         Run flake8"

install:
	$(PYTHON) -m pip install -r requirements.txt

train-smoke:
	$(PYTHON) src/modeling/train.py --data_dir $(DATA_DIR) --lr $(LR) --weights_path $(WEIGHTS_PATH) --onnx_path $(ONNX_PATH) --fast_dev_run

train:
	$(PYTHON) src/modeling/train.py --data_dir $(DATA_DIR) --lr $(LR) --weights_path $(WEIGHTS_PATH) --onnx_path $(ONNX_PATH)

visualize-network:
	$(PYTHON) src/modeling/train.py --data_dir $(DATA_DIR) --lr $(LR) --weights_path $(WEIGHTS_PATH) --visualize_network

predict:
	$(PYTHON) src/modeling/predict.py --weights_path $(WEIGHTS_PATH) --data_dir $(IMAGE_DIR) --predictions_path $(PREDICTIONS_PATH)

lint:
	$(PYTHON) -m flake8 src
