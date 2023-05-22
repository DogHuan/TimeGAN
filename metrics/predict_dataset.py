import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import  mean_squared_error, mean_absolute_error

from metrics.general_rnn import GeneralRNN

def predict_dataset(data, time):
    x = torch.tensor(data[:-1], dtype=torch.float32)
    t = torch.tensor(time[:-1], dtype=torch.float32)
    y = torch.tensor(data[1:], dtype=torch.float32)
    return torch.utils.data.TensorDataset(x, t, y)

def stock_predict(train_data, test_data):
    # 将train_data和test_data分别解包为数据和时间
    train_data, train_time = train_data
    test_data, test_time = test_data

    # 设置模型参数
    args = {}
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    args["task"] = "regression"
    args["model_type"] = "lstm"
    args["bidirectional"] = False
    args["epochs"] = 10
    args["batch_size"] = 128
    args["in_dim"] = train_data.shape[2]
    args["h_dim"] = train_data.shape[2]
    args["out_dim"] = train_data.shape[2]
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["max_seq_len"] = 10  # 只用于预测
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # 将训练数据和测试数据转换为数据集
    train_dataset = predict_dataset(train_data, train_time)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    test_dataset = predict_dataset(test_data, test_time)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_data), shuffle=False)

    # 初始化模型
    model = GeneralRNN(args)
    model.to(args["device"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

    # 训练模型
    for epoch in range(args["epochs"]):
        running_loss = 0.0
        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args["device"])
            train_y = train_y.to(args["device"])

            optimizer.zero_grad()
            train_p = model(train_x, train_t)
            loss = criterion(train_p, train_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # 测试模型
    model.eval()
    with torch.no_grad():
        rmse = 0
        mse = 0
        mae = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x, test_t).cpu()
            preds = test_p[:20, :1, :1].squeeze()
            truths = test_y[:20, :1, :1].squeeze()

            test_p = np.reshape(test_p.numpy(), [-1])
            test_y = np.reshape(test_y.numpy(), [-1])

            rmse += mean_squared_error(test_y, test_p, squared=False)
            mse += mean_squared_error(test_y, test_p)
            mae += mean_absolute_error(test_y, test_p)

            # Calculate the average of the RMSE, MSE, and MAE over all test batches
        n_batches = len(test_dataloader)
        rmse /= n_batches
        mse /= n_batches
        mae /= n_batches
        plt.plot(truths, label="True Values")
        plt.plot(preds, label="Predictions")
        plt.legend()
        plt.show()

    return rmse, mse, mae