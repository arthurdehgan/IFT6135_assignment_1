from itertools import product
import numpy as np
import torch
from torch import optim
from torch import nn
from network import Net, train, create_dataset


if __name__ == "__main__":
    test_path = "/home/arthur/git/IFT6135_assignment_1/testset/test/"
    dogs_path = "/home/arthur/git/IFT6135_assignment_1/trainset/Dog/"
    cats_path = "/home/arthur/git/IFT6135_assignment_1/trainset/Cat/"
    X, y, _ = create_dataset(cats_path, dogs_path, test_path)
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    best_vacc = 0
    LRS = [0.01, 0.005, 0.001]
    BATCH_SIZES = [32, 64, 128, 256]
    LAYERS = [128, 64, 32, 16]
    LINEARS = [128, 256, 512]
    for lin, lr, lay, bs in product(LINEARS, LRS, LAYERS, BATCH_SIZES):
        net = Net(3, lay, lin)
        print(net, lr, bs)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr)
        result = train(net, X, y, optimizer, criterion, bs, lr, p=25)

        if result["vacc"] > best_vacc:
            best_overall_model = result
            best_vacc = best_overall_model["vacc"]
    torch.save(best_overall_model["net"], "best_model_net")
    np.save("best_model_info", best_overall_model)
