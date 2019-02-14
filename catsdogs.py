from itertools import product
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.utils.data as utils
from params import batch_size, DATA_PATH


class Flatten(nn.Module):
    def forward(self, X):
        X = X.view(X.size(0), -1)
        return X


def accuracy(y_pred, target):
    correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
    return correct / len(target)


def evaluate(dataloader, batch_size=batch_size):
    LOSSES = 0
    ACCURACY = 0
    COUNTER = 0
    for batch in dataloader:
        optimizer.zero_grad()
        X, y = batch
        X = X.view(-1, 3, 64, 64)
        y = y.view(-1)
        X = X.cuda()
        y = y.cuda()

        loss = criterion(model(X), y)
        acc = accuracy(model(X), y)
        n = y.size(0)
        LOSSES += loss.sum().data.cpu().numpy() * n
        ACCURACY += acc.sum().data.cpu().numpy() * n
        COUNTER += n

    floss = LOSSES / float(COUNTER)
    faccuracy = ACCURACY / float(COUNTER)
    return floss, faccuracy


def train(dataloader, testloader, lr):
    GO = True
    STOP = 0
    EPOCH = 0
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    valid_loss = float("inf")
    best_model = {"vacc": 0}
    while GO:
        EPOCH += 1
        for batch in dataloader:
            optimizer.zero_grad()
            X, y = batch
            X = X.view(-1, 3, 64, 64)
            y = y.view(-1)
            X = X.cuda()
            y = y.cuda()

            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        old_valid_loss = valid_loss
        train_loss, train_acc = evaluate(dataloader)
        valid_loss, valid_acc = evaluate(validloader)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        if valid_acc > best_model["vacc"]:
            model.cpu()
            best_model = {
                "model": model,
                "epoch": EPOCH,
                "batch_size": batch_size,
                "lr": lr,
                "vacc": valid_acc,
                "vloss": valid_losses,
                "taccs": train_accs,
                "tloss": train_losses,
                "vaccs": valid_accs,
            }
            model.cuda()
        if valid_loss > old_valid_loss:
            STOP += 1
            if STOP > 2:
                GO = False
        else:
            STOP = 0

        print("EPOCH: {}".format(EPOCH))
        print(" [NLL] TRAIN {} / TEST {}".format(train_loss, valid_loss))
        print(" [ACC] TRAIN {} / TEST {}".format(train_acc, valid_acc))
    print("stopping, best valid acc is:", best_model["vacc"])
    return best_model


if __name__ == "__main__":
    X = np.load(DATA_PATH + "train_data.npy")
    X = X / np.max(X)
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = np.load(DATA_PATH + "train_target.npy")
    y = y[idx]
    # validation_set = np.load(DATA_PATH + "valid_data.npy")
    # validation = torch.Tensor(validation_set).float()

    X = torch.Tensor(X).float()
    y = torch.Tensor(y).long()
    train_size = int(0.8 * len(X))
    valid_size = len(X) - train_size
    train_index, valid_index = torch.utils.data.random_split(
        np.arange(len(X)), [train_size, valid_size]
    )
    train_dataset = utils.TensorDataset(X[train_index], y[train_index])
    valid_dataset = utils.TensorDataset(X[valid_index], y[valid_index])
    best_vacc = 0
    lrs = [0.05, 0.01, 0.005]
    batch_sizes = [64, 128]
    layers = [32, 16, 8]
    linears = [128, 256, 512]
    for lin, lr, lay, bs in product(linears, lrs, batch_sizes, layers):
        dataloader = utils.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        validloader = utils.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        model = nn.Sequential(
            nn.Conv2d(3, lay, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(lay, lay * 2, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(lay * 2, lay * 4, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(lay * 4, lay * 4, 3, 1),
            nn.MaxPool2d(3, 3),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(lay * 3 * 3 * 4, lin),
            nn.ReLU(True),
            nn.Linear(lin, 2),
        )

        model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print(model, lr, bs)
        result = train(dataloader, validloader, lr)
        if result["vacc"] > best_vacc:
            best_model = result
            best_vacc = best_model["vacc"]
    np.save("best_model", best_model)
