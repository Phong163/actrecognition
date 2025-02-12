import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siêu tham số
no_of_timesteps = 6
batch_size = 32
epochs = 16
learning_rate = 0.001

# Đọc dữ liệu từ folder
def read_data_from_folder(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dataset = df.iloc[:, :].values  # Bỏ cột đầu tiên nếu là index hoặc timestamp
            n_sample = len(dataset)
            for i in range(no_of_timesteps, n_sample):
                data.append((dataset[i-no_of_timesteps:i, :], label))
    return data

# Đọc dữ liệu từ các thư mục với 3 lớp
falling_data = read_data_from_folder("datasets/Falling", label=2)  # Lớp 2: Falling
normal_data = read_data_from_folder("datasets/Standing", label=0)   # Lớp 0: Normal
sitting_data = read_data_from_folder("datasets/Sitting", label=1) # Lớp 1: Sitting

# Trộn dữ liệu
all_data = falling_data + normal_data + sitting_data
np.random.shuffle(all_data)

# Tách X và y
X, y = zip(*all_data)
X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Cần dtype = long cho classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Tạo DataLoader
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Mô hình LSTM với PyTorch (3 output classes)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=4, num_classes=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)  # Output 3 lớp
        self.softmax = nn.Softmax(dim=1)  # Softmax cho multi-class classification
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Lấy đầu ra của timestep cuối cùng
        out = self.dropout(out)
        out = self.fc(out)
        return self.softmax(out)  # Softmax để tạo xác suất các lớp

# Khởi tạo mô hình
input_size = X.shape[2]  # Số feature trên mỗi timestep
model = LSTMModel(input_size).to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()  # Đổi sang CrossEntropyLoss() cho multi-class
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
def train_model(model, train_loader, test_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)  # Lấy nhãn dự đoán
            correct += (preds == y_batch).sum().item()
        
        acc = correct / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Chạy train
train_model(model, train_loader, test_loader, epochs)

# Lưu mô hình
torch.save(model.state_dict(), "model_lstm_multiclass.pth")
print("Model saved as model_lstm_multiclass.pth")
