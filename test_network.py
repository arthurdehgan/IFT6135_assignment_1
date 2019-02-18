import os
import numpy as np
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_valid_split
from matplotlib import img


def accuracy(y_pred, target):
    print(y_pred)
    y_pred_real = y_pred.max(1)[1]
    correct = torch.eq(y_pred_real, target).sum().type(torch.FloatTensor)
    return correct / len(target)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size, -1)


model = nn.Sequential()

model.add_module("conv1", nn.Conv2d(3, 32, 3, 1))
model.add_module("relu1", nn.ReLU())
model.add_module("maxp1", nn.MaxPool2d(2))
model.add_module("conv2", nn.Conv2d(32, 64, 3, 1))
model.add_module("relu2", nn.ReLU())
model.add_module("conv3", nn.Conv2d(64, 64, 3, 1))
model.add_module("relu3", nn.ReLU())
model.add_module("maxp2", nn.MaxPool2d(2))
model.add_module("flatten", Flatten())
model.add_module("maxp2", nn.Linear(338 * 32, 512))
model.add_module("relu4", nn.ReLU())
model.add_module("drop", nn.Dropout(inplace=True))
model.add_module("maxp2", nn.Linear(338 * 32, 512))
model.add_module("relu5", nn.ReLU())


def create_dataset(cats_path, dogs_path, test_path):
    train_set = []
    target = []
    n_cats = len(os.listdir(cats_path))
    for i in range(1, n_cats):
        f = "{}.Cat.jpg".format(i)
        mat = img.imread(cats_path + f)
        if not len(mat.shape) == 3:
            mat = np.array((mat, mat, mat)).T
        train_set.append(mat)
        target.append(0)

    n_dogs = len(os.listdir(dogs_path))
    for i in range(1, n_dogs):
        f = "{}.Dog.jpg".format(i)
        mat = img.imread(dogs_path + f)
        if not len(mat.shape) == 3:
            mat = np.array((mat, mat, mat)).T
        train_set.append(mat)
        target.append(1)

    train_set = np.asarray(train_set)
    target = np.asarray(target)

    test_set = []
    n_valid = len(os.listdir(test_path))
    for i in range(1, n_valid):
        f = "{}.jpg".format(i)
        mat = img.imread(test_path + f)
        if not len(mat.shape) == 3:
            mat = np.array((mat, mat, mat)).T
        test_set.append(mat)

    return train_set / 255.0, target, np.asarray(test_set) / 255.0


test_path = "/home/arthur/git/IFT6135_assignment_1/testset/test/"
dogs_path = "/home/arthur/git/IFT6135_assignment_1/trainset/Dog/"
cats_path = "/home/arthur/git/IFT6135_assignment_1/trainset/Cat/"

X, y, sub = create_dataset(cats_path, dogs_path, test_path)

idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

X_train, X_valid, y_train, y_valid = train_valid_split(
    X, y, test_size=.25, shuffle=True, stratify=y, random_state=420
)

X_train = torch.Tensor(X_train).float()
y_train = torch.Tensor(y_train).float()
X_valid = torch.Tensor(X_valid).float()
y_valid = torch.Tensor(y_valid).float()

batch_size = 500
p = 10
j = 0
epoch = 0
best_loss = float("inf")
optimizer = optim.SGD(model.parameters(), lr=0.0005)
while j < p:
    epoch += 1
    tacc = []
    tloss = []
    for i in range(0, len(X_train), batch_size):
        y_batch = y[i : i + batch_size]
        y_pred = model(X_train[i : i + batch_size])
        loss = nn.CrossEntropyLoss()
        tacc.append(accuracy(y_pred, y_batch))
        optimizer.zero_grad()
        tloss = loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    y_pred_valid = model(X_valid)
    vacc = accuracy(y_pred_valid, y_valid)
    vloss = loss(y_pred_valid, y_valid)

    if vloss < best_loss:
        j = 0
        best_loss = vloss
    else:
        j += 1

    print(epoch)
    print("TACC {}, TLOSS {}".format(np.mean(tacc), np.mean(tloss)))
    print("VACC {}, VLOSS {}".format(vacc, vloss))
