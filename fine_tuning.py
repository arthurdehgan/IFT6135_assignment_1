from itertools import product
import numpy as np
import torch
from torch import optim
from torch import nn
from network import Net, train


if __name__ == "__main__":
    DATA_PATH = "./"
    X = np.load(DATA_PATH + "train_data.npy")
    y = np.load(DATA_PATH + "train_target.npy")
    X = X / np.max(X)
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    best_vacc = 0
    LRS = [0.05]
    BATCH_SIZES = [64, 128]
    LAYERS = [128, 64, 32, 16]
    LINEARS = [128, 256]
    for lin, lr, lay, bs in product(LINEARS, LRS, LAYERS, BATCH_SIZES):
        net = Net(3, lay, lin)
        print(net, lr, bs)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr)
        result = train(net, X, y, optimizer, criterion, bs, lr)

        if result["vacc"] > best_vacc:
            best_overall_model = result
            best_vacc = best_overall_model["vacc"]
    torch.save(best_overall_model["net"], "best_model_net")
    np.save("best_model_info", best_overall_model)
