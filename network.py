import numpy as np
import torch
from torch import nn
import torch.utils.data as utils


def train(net, X, y, optimizer, criterion, batch_size, lr, p=30):
    X = torch.Tensor(X).float()
    N = len(X)
    y = torch.Tensor(y).long()
    train_size = int(0.75 * N)
    valid_size = N - train_size
    train_index, valid_index = torch.utils.data.random_split(
        np.arange(N), [train_size, valid_size]
    )
    train_dataset = utils.TensorDataset(X[train_index], y[train_index])
    valid_dataset = utils.TensorDataset(X[valid_index], y[valid_index])
    dataloader = utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validloader = utils.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    j = 0
    epoch = 0
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    best_vloss = float("inf")
    while j < p:
        epoch += 1
        net.train_mode()
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

        net.eval_mode()
        train_loss, train_acc = net.evaluate(dataloader, criterion)
        valid_loss, valid_acc = net.evaluate(validloader, criterion)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        if valid_loss < best_vloss:
            best_vacc = valid_acc
            best_vloss = valid_loss
            best_net = net
            best_epoch = epoch
            j = 0
        else:
            j += 1

        print("epoch: {}".format(epoch))
        print(" [LOSS] TRAIN {} / VALID {}".format(train_loss, valid_loss))
        print(" [ACC] TRAIN {} / VALID {}".format(train_acc, valid_acc))
    best_model = {
        "net": best_net.state_dict(),
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "batch_size": batch_size,
        "lr": lr,
        "vacc": best_vacc,
        "vloss": best_vloss,
        "taccs": train_accs,
        "tloss": train_losses,
        "vaccs": valid_accs,
        "convs": net.conv_size,
        "lins": net.lin_size,
    }
    return best_model


def accuracy(y_pred, target):
    correct = torch.eq(y_pred.max(1)[1], target).sum().type(torch.FloatTensor)
    return correct / len(target)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Dropout(nn.Module):
    def __init__(self, p=0.3):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        data = x.data
        shape = data.shape
        size = int(shape[0] * shape[1])
        drop_idx = np.random.choice(
            np.arange(size), replace=False, size=int(size * self.p)
        )
        data = data.flatten()
        data[drop_idx] = 0
        data = data.reshape(shape)
        x.data = data
        return x


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
        self.dropout = nn.Sequential(Dropout()).cuda()
        self.lin = nn.Sequential(nn.ReLU(True), nn.Linear(lin_size, 2)).cuda()

    def eval_mode(self):
        self.drop = False

    def train_mode(self):
        self.drop = True

    def forward(self, x):
        dat = self.model(x)
        if self.drop:
            dat = self.dropout(dat)
        return self.lin(dat)

    def evaluate(self, dataloader, criterion):
        self.eval_mode()
        LOSSES = 0
        ACCURACY = 0
        COUNTER = 0
        for batch in dataloader:
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
    
    def train(self, X, y, epochs, optimizer, criterion, batch_size, lr):
        train_dataset = utils.TensorDataset(X, y)
        dataloader = utils.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        train_accs = []
        train_losses = []
        for e in epochs:
            net.train_mode()
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

            net.eval_mode()
            train_loss, train_acc = net.evaluate(dataloader, criterion)

            train_accs.append(train_acc)
            train_losses.append(train_loss)

            print("epoch: {}".format(epoch))
            print("TRAIN : [LOSS] {} / [ACC] {}".format(train_loss, train_acc))
        print('Training Done')

