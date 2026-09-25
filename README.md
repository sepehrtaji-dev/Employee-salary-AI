# Employee Salary AI

A **PyTorch MLP** regression model that predicts employee salary from HR features.

## Model

| Component | Details |
|-----------|---------|
| Architecture | MLP: 128 → 64 → 32 → 1 |
| Loss | MSELoss |
| Optimizer | Adam (lr=0.001) |
| Batch size | 64 |
| Epochs | 100 |
| Split | 70% train / 30% test |

Inputs and outputs are both StandardScaler normalized. Saved scalers (`x_scaler.pkl`, `y_scaler.pkl`) allow inverse-transforming predictions back to real salary values.

## Setup

```bash
pip install torch pandas scikit-learn matplotlib joblib
```

## Run

```bash
python model.py
```

Saves model to `salary_model_best.pth`.

## Project Structure

```
Employee-salary-AI/
├── model.py              # Model + training
├── df_builder.py         # Dataset builder
├── employee_salary.csv   # Dataset
├── salary_model_best.pth # Saved weights
├── x_scaler.pkl          # Feature scaler
└── y_scaler.pkl          # Target scaler
```
