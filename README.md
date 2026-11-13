# Certainty Pooling for Multiple Instance Learning

PyTorch implementation of **Certainty Pooling** ([Gildenblat et al., 2020](https://arxiv.org/abs/2008.10548)) for weakly supervised classification of histopathology slides: one label per slide, no label per tile.

## Problem

Each slide is a *bag* of tiles. Tiles are represented by pre-extracted ResNet features (one `.npy` file per slide), and only the slide-level binary label is known. The model has to score a slide from its tiles without tile-level supervision.

## Method

Max pooling picks the tile with the highest score, so a single over-confident negative tile can flip the slide prediction. Certainty pooling weights each tile score by the model's certainty about it:

1. Run `T` stochastic forward passes with dropout active (Monte-Carlo dropout).
2. Certainty of tile `k`: `C_k = 1 / (std(X_k) + eps)`, where `X_k` is the vector of its `T` predictions.
3. The slide is represented by the tile `k* = argmax_k (C_k * h_k)`, where `h_k` is the deterministic prediction.
4. The loss is computed on the selected tile only.

## What is in the repository

| Path | Content |
|---|---|
| `src/models/certainty_pooling.py` | Certainty pooling, per bag and batched over bags of different sizes |
| `src/models/model.py` | Tile classifier: 5 fully connected layers with dropout, sigmoid output |
| `src/models/losses.py` | Class-weighted binary cross-entropy |
| `src/models/training.py`, `evaluation.py` | Training loop and validation metrics (classification report, confusion matrix) |
| `src/training/` | Single run, repeated runs and stratified k-fold training |
| `src/submission/` | Inference on a held-out set, single model or k-fold ensemble |
| `src/config/training_config.yaml` | Hyper-parameters (learning rate, epochs, batch size, number of MC passes) |

## Usage

Requirements: Python 3.10, [Poetry](https://python-poetry.org/).

```bash
poetry install
```

Set `DATA_PATH` in `src/config/settings.py` and the paths in `src/config/training_config.yaml`, then:

```bash
# single training run
poetry run python src/models/main.py --config-path src/config/training_config.yaml

# stratified k-fold
poetry run python src/training/k_fold_training.py
```

Expected data layout:

```
<DATA_PATH>/
  train_output.csv              # columns: ID, Target
  train_input/resnet_features/  # <ID>.npy, one row per tile
  test_input/resnet_features/
```

The dataset is not distributed with this repository.

## Reference

J. Gildenblat, I. Ben-Shaul, Z. Lapp, E. Klaiman. *Certainty Pooling for Multiple Instance Learning*. arXiv:2008.10548, 2020.
