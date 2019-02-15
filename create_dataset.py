import numpy as np
from matplotlib import image as img
import os

cats = "trainset/Cat/"
dogs = "trainset/Dog/"

train_set = []
target = []
for f in os.listdir(cats):
    if f.endswith("jpg"):
        mat = img.imread(cats + f)
        if not len(mat.shape) == 3:
            mat = np.array((mat, mat, mat)).T
        train_set.append(mat)
        target.append(0)

for f in os.listdir(dogs):
    if f.endswith("jpg"):
        mat = img.imread(dogs + f)
        if not len(mat.shape) == 3:
            mat = np.array((mat, mat, mat)).T
        train_set.append(mat)
        target.append(1)

train_set = np.asarray(train_set)
target = np.asarray(target)
print(train_set.shape, target.shape)
np.save("train_data", train_set)
np.save("train_target", target)
del train_set

test_set = []
for i in range(1, 5000):
    f = "testset/test/{}.jpg".format(i)
    mat = img.imread(f)
    if not len(mat.shape) == 3:
        mat = np.array((mat, mat, mat)).T
    test_set.append(mat)

print(len(test_set))
np.save("test_data", np.array(test_set))
