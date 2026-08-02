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
test_loader = DataLoader(test_ds, shuffle=False, batch_size=64)

class salary_classification(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

epochs = 100
train_losses = []
eval_losses = []
model = salary_classification(x_train_tensor.shape[1])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    train_epoch_losses = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)  
        batch_y = batch_y.to(device)
        y_pred = model(batch_x)
        train_loss = loss_fn(y_pred, batch_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_epoch_losses += train_loss.item()
    train_epoch_losses = train_epoch_losses / len(train_loader)
    train_losses.append(train_epoch_losses)

    model.eval()
    test_epoch_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)  
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            test_loss = loss_fn(y_pred, batch_y)
            test_epoch_loss += test_loss.item()
        test_epoch_loss = test_epoch_loss/len(test_loader)
        eval_losses.append(test_epoch_loss)
    if epoch%5 == 0:
        print(f"epoch : {epoch}, train loss : {train_loss}, test loss : {test_loss}")

plt.plot(range(epochs), train_losses, label = "train")
plt.plot(range(epochs), eval_losses, label = "test")
plt.title("test train losses")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()