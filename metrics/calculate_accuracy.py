import numpy as np
import torch
from sklearn import metrics
from tqdm import trange

from metrics.dataset import OneStepPredictionDataset
from metrics.general_rnn import GeneralRNN


def accuracy2(pred_choice, target):
    TP, TN, FN, FP = 0, 0, 0, 0
    # TP    predict 和 label 同时为1
    TP += ((pred_choice == 1) & (target.data == 1)).cpu().sum()
    # TN    predict 和 label 同时为0
    TN += ((pred_choice == 0) & (target.data == 0)).cpu().sum()
    # FN    predict 0 label 1
    FN += ((pred_choice == 0) & (target.data == 1)).cpu().sum()
    # FP    predict 1 label 0
    FP += ((pred_choice == 1) & (target.data == 0)).cpu().sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return F1, acc


def CalculateAccuracy(train_data, test_data):
    train_data, train_time = train_data
    test_data, test_time = test_data

    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters
    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 20
    args["batch_size"] = 128
    args["in_dim"] = dim
    args["h_dim"] = dim
    args["out_dim"] = dim
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["padding_value"] = 0.0
    args["max_seq_len"] = 100 - 1  # only 99 is used for prediction
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Set training features and labels
    train_dataset = OneStepPredictionDataset(train_data, train_time)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True
    )

    # Set testing features and labels
    test_dataset = OneStepPredictionDataset(test_data, test_time)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=no,
        shuffle=True
    )
    # Initialize model
    model = GeneralRNN(args)
    model.to(args["device"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["learning_rate"]
    )

    # Train the predictive model
    logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0

        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args["device"])
            train_y = train_y.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

    # Evaluate the trained model
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x, test_t).cpu()
            test_p = torch.round(test_p).cpu().numpy().astype(int)
            test_p = np.reshape(test_p, [-1])
            test_y = np.reshape(test_y.numpy(), [-1])
            labels_all = np.append(labels_all, test_y)
            predict_all = np.append(predict_all, test_p)

    accuracy = metrics.accuracy_score(labels_all, predict_all)

    return accuracy
