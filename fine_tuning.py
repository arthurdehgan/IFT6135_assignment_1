from itertools import product
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.utils.data as utils


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Dropout(nn.Module):
    def __init__(self, p=0.3):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        data = x.data.cpu().numpy()
        shape = data.shape
        drop_idx = np.random.choice(
            np.arange(data.size), replace=False, size=int(data.size * self.p)
        )
        data = data.flatten()
        data[drop_idx] = 0
        data = data.reshape(shape)
        x.data = torch.from_numpy(data).cuda()
        return x


def accuracy(y_pred, target):
    correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
    return correct / len(target)


def train(net, dataloader, validloader, lr, p=50):
    EPOCH = 0
    j = 0
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    best_vloss = float("inf")
    while j < p:
        EPOCH += 1
        net.train()
        for batch in dataloader:
            optimizer.zero_grad()
            X, y = batch
            X = X.view(-1, 3, 64, 64)
            y = y.view(-1)
            X = X.cuda()
            y = y.cuda()

            loss = criterion(net.forward(X), y)
            loss.backward()
            optimizer.step()

        net.eval()
        train_loss, train_acc = net.evaluate(dataloader)
        valid_loss, valid_acc = net.evaluate(validloader)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        if valid_loss < best_vloss:
            best_model = {
                "net": net.state_dict(),
                "epoch": EPOCH,
                "batch_size": batch_size,
                "lr": lr,
                "vacc": valid_acc,
                "vloss": valid_losses,
                "taccs": train_accs,
                "tloss": train_losses,
                "vaccs": valid_accs,
                "convs": net.conv_size,
                "lins": net.lin_size,
            }
            best_vloss = valid_loss
        else:
            j += 1

        print("EPOCH: {}".format(EPOCH))
        print(" [LOSS] TRAIN {} / TEST {}".format(train_loss, valid_loss))
        print(" [ACC] TRAIN {} / TEST {}".format(train_acc, valid_acc))
    return best_model


class Net(nn.Module):
    def __init__(self, input_size, conv_size, lin_size):
        super(Net, self).__init__()

        self.drop = True
        self.lin_size = lin_size
        self.conv_size = conv_size
        self.model = nn.Sequential(
            nn.Conv2d(input_size, conv_size, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(conv_size, conv_size * 2, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(conv_size * 2, conv_size * 4, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(conv_size * 4, conv_size * 4, 3, 1),
            nn.MaxPool2d(3, 3),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(conv_size * 3 * 3 * 4, lin_size),
        ).cuda()
        self.lin = nn.Sequential(nn.ReLU(True), nn.Linear(lin_size, 2)).cuda()
        self.dropout = nn.Sequential(Dropout()).cuda()

    def eval(self):
        self.drop = False

    def train(self):
        self.drop = True

    def forward(self, x):
        dat = self.model(x)
        if self.drop:
            dat = self.dropout(dat)
        return self.lin(dat)

    def evaluate(self, dataloader):
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

            loss = criterion(self.forward(X), y)
            acc = accuracy(self.forward(X), y)
            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            ACCURACY += acc.sum().data.cpu().numpy() * n
            COUNTER += n

        floss = LOSSES / float(COUNTER)
        faccuracy = ACCURACY / float(COUNTER)
        return floss, faccuracy


if __name__ == "__main__":
    DATA_PATH = "./"
    X = np.load(DATA_PATH + "train_data.npy")
    y = np.load(DATA_PATH + "train_target.npy")
    X = X / np.max(X)
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    X = torch.Tensor(X).float()
    N = len(X)
    y = torch.Tensor(y).long()
    train_size = int(0.8 * N)
    valid_size = int((N - train_size) / 2.0)
    test_size = N - train_size - valid_size
    train_index, remaining_index = torch.utils.data.random_split(
        np.arange(N), [train_size, N - train_size]
    )
    valid_index, test_index = torch.utils.data.random_split(
        np.arange(N - train_size), [valid_size, test_size]
    )
    train_dataset = utils.TensorDataset(X[train_index], y[train_index])
    valid_dataset = utils.TensorDataset(X[valid_index], y[valid_index])
    test_dataset = utils.TensorDataset(X[test_index], y[test_index])

    batch_size = 64

    best_vacc = 0
    lrs = [0.05]
    batch_sizes = [64, 128]
    layers = [128, 64, 32, 16]
    linears = [128, 256]
    for lin, lr, lay, bs in product(linears, lrs, layers, batch_sizes):
        dataloader = utils.DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=2
        )
        validloader = utils.DataLoader(
            valid_dataset, batch_size=bs, shuffle=True, num_workers=2
        )
        testloader = utils.DataLoader(
            test_dataset, batch_size=bs, shuffle=True, num_workers=2
        )
        net = Net(3, lay, lin)
        print(net, lr, bs)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr)
        result = train(net, dataloader, validloader, lr)
        model = Net(3, result["convs"], result["lins"])
        model.load_state_dict(result["net"])
        print("Test accuracy is", model.evaluate(testloader))

        if result["vacc"] > best_vacc:
            best_overall_model = result
            best_vacc = best_overall_model["vacc"]
    torch.save(best_overall_model["net"], "best_model_net")
    np.save("best_model_info", best_overall_model)