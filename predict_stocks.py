import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def predict_stocks(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将数据转换为 PyTorch 张量
    data = torch.tensor(data, dtype=torch.float32).to(device)

    # 定义模型超参数
    input_size = data.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = 2

    # 初始化模型并将其移动到GPU（如果可用）

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.to(device)

    # 加载训练好的模型参数（可选）
    # model.load_state_dict(torch.load("model.pt"))

    # 对数据进行预测
    model.eval()
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        outputs = model(data)
        _, predicted_labels = torch.max(outputs, 1)
        label = data[:, 0, 0]
        true_labels.extend(label.tolist())

    # 计算评估指标

    true_labels = np.round(true_labels)
    predicted_labels = predicted_labels.cpu().numpy()
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    return accuracy, precision, recall
