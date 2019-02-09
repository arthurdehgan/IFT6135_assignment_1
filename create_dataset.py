import numpy as np
from matplotlib import image as img
from path import Path as path
from params import TRAIN_SET, TEST_SET

Cats = path("/home/arthur/data/Kaggle/trainset/Cat")
Dogs = path("/home/arthur/data/Kaggle/trainset/Dog")
TEST_SET = path(TEST_SET)
TRAIN_SET = path(TRAIN_SET)

train_set = []
target = []
for f in Cats.listdir():
    mat = img.imread(f)
    if not np.isnan(mat).any() and len(mat.shape) == 3:
        train_set.append(mat)
        target.append(0)

for f in Dogs.listdir():
    mat = img.imread(f)
    if not np.isnan(mat).any() and len(mat.shape) == 3:
        train_set.append(mat)
        target.append(1)

train_set = np.asarray(train_set)
target = np.asarray(target)
print(train_set.shape, target.shape)
np.save(TRAIN_SET.parent / "train_data", train_set)
np.save(TRAIN_SET.parent / "train_target", target)
del train_set

# test_set = []
# for f in (TEST_SET / "test").listdir():
#     mat = img.imread(f)
#     if not np.isnan(mat).any() and len(mat.shape) == 3:
#         test_set.append(mat)
#
# np.save(TEST_SET.parent / "test_data", np.array(test_set))
