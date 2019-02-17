import numpy as np
import torch
from torch import nn
from torch import optim
import torch.utils.data as utils


def train(net, X, y, batch_size, lr, p=15, rep=5):
    X = torch.Tensor(X).float()
    N = len(X)
    y = torch.Tensor(y).long()
    train_size = int(0.9 * N)
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
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    epoch_list = [0]
    best_vloss = float("inf")
    for i in range(rep):
        print("Training with batch_size: {}, learning_rate: {}".format(batch_size, lr))
        optimizer = optim.SGD(net.parameters(), lr=lr)
        j = 0
        epoch = epoch_list[-1]
        if i >= 1:
            net.load_state_dict(best_state)
        while j < p:
            epoch += 1
            print("epoch: {}".format(epoch))
            net.train_mode()
            for batch in dataloader:
                optimizer.zero_grad()
                X, y = batch
                X = X.view(-1, 3, 64, 64)
                y = y.view(-1)
                X = X.cuda()
                y = y.cuda()

                loss = nn.CrossEntropyLoss()(net.forward(X), y)
                loss.backward()
                optimizer.step()

            net.eval_mode()
            train_loss, train_acc = net.evaluate(dataloader)
            valid_loss, valid_acc = net.evaluate(validloader)

            train_accs.append(train_acc)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            if valid_loss < best_vloss:
                print("Best epoch yet")
                best_vacc = valid_acc
                best_vloss = valid_loss
                best_state = net.state_dict()
                best_epoch = epoch
                j = 0
            else:
                j += 1

            print(" [LOSS] TRAIN {} / VALID {}".format(train_loss, valid_loss))
            print(" [ACC] TRAIN {} / VALID {}".format(train_acc, valid_acc))
        epoch_list.append(best_epoch)
        batch_size = int(batch_size / 2)
        if int(lr * 10000) % 5:
            lr = lr / 5.0
        else:
            lr = lr / 2.0
    best_model = {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "epoch_list": epoch_list,
        "real_epochs": [e + i * p for i, e in enumerate(epoch_list)],
        "batch_size": batch_size,
        "lr": lr,
        "vacc": best_vacc,
        "best_vloss": best_vloss,
        "vloss": valid_losses,
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
    def __init__(self, p=0.5):
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
            nn.ReLU(True),
            nn.Conv2d(conv_size * 2, conv_size * 2, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(conv_size * 338, lin_size),
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

    def evaluate(self, dataloader):
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

            loss = nn.CrossEntropyLoss()(self.forward(X), y)
            acc = accuracy(self.forward(X), y)
            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            ACCURACY += acc.sum().data.cpu().numpy() * n
            COUNTER += n

        floss = LOSSES / float(COUNTER)
        faccuracy = ACCURACY / float(COUNTER)
        return floss, faccuracy

    def train(self, X, y, epochs, batch_size, lr):
        X = torch.Tensor(X).float()
        y = torch.Tensor(y).long()
        train_dataset = utils.TensorDataset(X, y)
        dataloader = utils.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        train_accs = []
        train_losses = []
        for i in range(epochs[1:]):
            optimizer = optim.SGD(self.parameters(), lr=lr)
            for e in range(epochs[i - 1], epochs[i]):
                self.train_mode()
                for batch in dataloader:
                    optimizer.zero_grad()
                    X, y = batch
                    X = X.view(-1, 3, 64, 64)
                    y = y.view(-1)
                    X = X.cuda()
                    y = y.cuda()

                    loss = nn.CrossEntropyLoss()(self.forward(X), y)
                    loss.backward()
                    optimizer.step()

                self.eval_mode()
                train_loss, train_acc = self.evaluate(dataloader)

                train_accs.append(train_acc)
                train_losses.append(train_loss)

                print("epoch: {}".format(e))
                print("TRAIN : [LOSS] {} / [ACC] {}".format(train_loss, train_acc))
            batch_size = int(batch_size / 2)
            if int(lr * 10000) % 5:
                lr = lr / 5.0
            else:
                lr = lr / 2.0
        print("Training Done")
