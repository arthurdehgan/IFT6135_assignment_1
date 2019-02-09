import math
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.utils.data as utils
from params import batch_size, store_every, n_epochs, DATA_PATH, learning_rate
from sklearn.model_selection import train_test_split


class Flatten(nn.Module):
    def forward(self, X):
        X = X.view(X.size(0), -1)
        return X


class Dropout(nn.Module):
    def forward(self, X, p=0.2):
        shape = X.shape
        size = int(p * np.prod(shape))
        drop_idx = np.random.choice(np.arange(X.size), replace=False, size=size)
        X = X.flatten()
        X[drop_idx] = 0
        return X.reshape(shape)


class NN(nn.Module):
    def __init__(self, features):
        super(NN, self).__init__()

        self.features = features
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, X):
        X = self.features(X)
        return X


model = nn.Sequential(
    nn.Conv2d(3, 64, 3, 1),
    nn.ReLU(True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, 3, 1),
    nn.ReLU(True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(128, 256, 3, 1),
    nn.MaxPool2d(2, 2),
    nn.ReLU(True),
    nn.Conv2d(256, 512, 3, 1),
    nn.MaxPool2d(2, 2),
    nn.ReLU(True),
    Flatten(),
    nn.Linear(2048, 512),
    nn.ReLU(True),
    nn.Linear(512, 512),
    nn.ReLU(True),
    nn.Linear(512, 2),
)

model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


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


def train(dataloader, testloader):
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    avg_loss = float("inf")
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    for e in range(n_epochs):
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

            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            COUNTER += n
            ITERATIONS += 1
            if ITERATIONS % (store_every / 2) == 0:
                avg_loss = LOSSES / float(COUNTER)
                LOSSES = 0
                COUNTER = 0
                print(" Iteration {}: TRAIN {}".format(ITERATIONS, avg_loss))

            if ITERATIONS % (store_every) == 0:

                train_loss, train_acc = evaluate(dataloader)
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                test_loss, test_acc = evaluate(testloader)
                test_accs.append(test_acc)
                test_losses.append(test_loss)

                print(" [EPOCH] {}".format(e + 1))
                print(" [NLL] TRAIN {} / TEST {}".format(train_loss, test_loss))
                print(" [ACC] TRAIN {} / TEST {}".format(train_acc, test_acc))


if __name__ == "__main__":
    X = np.load(DATA_PATH + "train_data.npy")
    X = X / np.max(X)
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = np.load(DATA_PATH + "train_target.npy")
    y = y[idx]
    # validation_set = np.load(DATA_PATH + "test_data.npy")
    # validation = torch.Tensor(validation_set).float()

    X = torch.Tensor(X).float()
    y = torch.Tensor(y).long()
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size
    train_index, test_index = torch.utils.data.random_split(
        np.arange(len(X)), [train_size, test_size]
    )
    train_dataset = utils.TensorDataset(X[train_index], y[train_index])
    dataloader = utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_dataset = utils.TensorDataset(X[test_index], y[test_index])
    testloader = utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    train(dataloader, testloader)
