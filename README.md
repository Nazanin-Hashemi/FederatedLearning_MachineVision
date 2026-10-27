# Federated Learning with Sparse Task Arithmetic on CIFAR-100

This repository implements a Federated Learning (FedAvg) with sparse fine-tuning and task arithmetic, using a DINO-pretrained Vision Transformer backbone on CIFAR-100.

## Features

- Centralized baseline training
- Federated learning (IID / non-IID)
- Fisher Information–based sparse fine-tuning
- Task arithmetic with scalable task vectors
- Fully reproducible experiments via Hydra configuration



##  Project Structure

```text
.
├── configs/                    # Hydra YAML configurations
│   ├── base.yaml
│   ├── centralized.yaml
│   └── fedavg.yaml
│
├── src/                        # Core implementation
│   ├── datamodule.py           # Centralized & federated data handling
│   ├── model.py                # DINO + linear classifier
│   ├── federated.py            # FedAvg implementation
│   ├── pruner.py               # Fisher-based mask calibration
│   ├── sparse_optimizer.py     # SparseSGDM optimizer
│   ├── train.py                # Centralized training script
│   ├── train_fedavg.py         # Federated training script
│   └── eval_task_arithmetic.py # Task arithmetic evaluation
│
├── notebooks/
│   ├── 00_centralized_baseline.ipynb
│   ├── 01_federated_baseline.ipynb
│   └── 02_task_arithmetic.ipynb
│
├── results/                    # Logs, checkpoints, outputs
├── requirements.txt
└── README.md
```


##  Environment Setup


### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

###  Dataset

The DINO backbone is loaded via torch.hub and requires an active internet connection on first run. The project uses CIFAR-100 via torchvision. The dataset is downloaded automatically. Default location: ./data and No manual download is required.


## Experiments and Reproducibility

All experiments are controlled via Hydra YAML configurations located in configs/.

Configuration hierarchy:

- base.yaml → shared settings

- centralized.yaml → centralized baseline

- fedavg.yaml → federated learning + sparse task arithmetic

## Centralized Baseline Training

This experiment trains a centralized classifier on CIFAR-100 and serves as the baseline model.

Run command
```bash
python -m src.train
```


Configuration used: configs/centralized.yaml that automatically loads base.yaml


## Outputs

- Best model checkpoint (by validation accuracy)
- Training and validation logs
- Test accuracy of the best checkpoint

## Purpose

- Reference model for comparison with federated learning

- Source of the base model in task arithmetic

## Federated Learning (FedAvg)

This experiment runs Federated Averaging with configurable client participation, local steps, and data sharding.

Run command
```bash
python -m src.train_fedavg
```


Configuration used:

- configs/fedavg.yaml
- Automatically loads base.yaml


Key configurable parameters:

- num_clients
- participation_rate
- local_steps
- num_rounds
- sharding (iid or non_iid)

Outputs:

- Federated model checkpoints
- Per-round validation accuracy
- Final test accuracy
- JSON file with full experiment history


## Sparse Fine-Tuning (Fisher-Based)

Sparse fine-tuning is enabled inside federated learning by setting:
```yaml
federated:
  use_sparse: true
```


Process:
- Ridge initialization of the classifier head
- Fisher Information calibration on validation data
- Selection of least-sensitive parameters
- Sparse updates using SparseSGDM

```yaml
Relevant config fields
use_sparse: true
sparsity_level: 0.5
num_calibration_rounds: 3
sparse_strategy: least_sensitive
```

4️⃣ Task Arithmetic Evaluation

This step evaluates linear combinations of the centralized and federated models.

Run command
```yaml
python -m src.eval_task_arithmetic
```

What it does:
- Loads the centralized baseline model
- Loads the federated model
- Sweeps over multiple values of α
- Evaluates each combined model on the test set

Output:

- Table of test accuracies for different α values
- Identification of optimal task arithmetic scaling

## How to Reproduce the Results

Follow these steps to fully reproduce the experiments reported in this work.

### Centralized baseline
```bash
python -m src.train
```


### Federated learning + sparse fine-tuning
```bash
python -m src.train_fedavg
```


All results are deterministic given the fixed seed defined in configs/base.yaml.