# House Price Prediction

This project builds a house price prediction model using the California housing dataset in `housing.csv`. The main script trains a `RandomForestRegressor` with preprocessing steps for numeric and categorical features, then saves the trained model and preprocessing pipeline for later reuse.

## Project Structure

- `main.py` - trains the model on first run, then performs inference on later runs
- `data_preprocessing.py` - standalone preprocessing and model comparison script
- `check_accuracy.py` - helper script for checking model performance
- `housing.csv` - source dataset used for training and evaluation
- `input.csv` - generated test/input data created during training
- `output.csv` - prediction output generated during inference

## Requirements

- Python 3.9 or newer
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`

Install the dependencies with:

```bash
pip install pandas numpy scikit-learn joblib
```

## How It Works

The workflow in `main.py` is:

1. If `model.pkl` does not exist, the script trains a model from `housing.csv`.
2. The training step also saves `pipeline.pkl` and creates `input.csv` from the held-out test split.
3. If `model.pkl` already exists, the script loads `model.pkl` and `pipeline.pkl`, reads `input.csv`, and writes predictions to `output.csv`.

Because the model artifacts are generated files, they are ignored by git and should not be committed.

## Run the Project

### First run: train the model

```bash
python main.py
```

This creates:

- `model.pkl`
- `pipeline.pkl`
- `input.csv`

### Later runs: generate predictions

Run the same command again:

```bash
python main.py
```

If the saved artifacts are present, the script performs inference and writes `output.csv`.

## Output

The prediction file contains the original input columns plus a new column named `median_house_value_predicted`.

## Notes

- The first run must have access to `housing.csv`.
- If you want to retrain from scratch, delete `model.pkl` and `pipeline.pkl` before running `main.py` again.
- `data_preprocessing.py` is useful if you want to experiment with the preprocessing pipeline or compare different regressors.
