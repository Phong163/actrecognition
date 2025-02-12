import torch
import torch.nn as nn
import torch.optim as optim

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
    
    