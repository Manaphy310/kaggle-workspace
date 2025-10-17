# Kaggle Competition Project

## Project Structure

```
.
├── data/
│   ├── raw/          # Original competition data
│   └── processed/    # Processed and feature-engineered data
├── notebooks/        # Jupyter notebooks for EDA and experiments
├── src/              # Source code for reproducible pipeline
│   ├── features/     # Feature engineering code
│   ├── models/       # Model training and prediction code
│   └── utils/        # Utility functions
├── models/           # Trained model files
└── submissions/      # Submission files
```

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (install via: `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Download competition data:
```bash
uv run kaggle competitions download -c <competition-name>
unzip <competition-name>.zip -d data/raw/
```

## Usage

### Running Jupyter Notebook

```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

### Running Python scripts

```bash
uv run python src/models/train.py
```

### Adding new dependencies

```bash
# Add a package
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Add optional packages (e.g., XGBoost, LightGBM, PyTorch)
uv add xgboost lightgbm
```

## Notes

TBD
