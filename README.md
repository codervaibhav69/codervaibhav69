# House Price Prediction (Linear Regression)

This project trains a simple linear regression model to predict house prices from numeric features.

## Project Structure

- `model.py`: Loads data, trains a linear regression model, and prints evaluation metrics.
- `data/house_prices.csv`: Sample dataset with basic house features.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
python model.py --data data/house_prices.csv
```

## Dataset Columns

- `square_feet`
- `bedrooms`
- `bathrooms`
- `age`
- `price` (target)

Feel free to replace `data/house_prices.csv` with your own dataset as long as it contains the same columns.
