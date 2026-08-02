import pandas as pd
import torch.nn as nn
import  torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv("employee_salary.csv")
x = df.iloc[:, :8]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=.3, random_state=42)
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32, device=device)
train_ds = TensorDataset(x_train_tensor, y_train_tensor)
test_ds = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_ds = DataLoader(test_ds, shuffle=False, batch_size=64)

