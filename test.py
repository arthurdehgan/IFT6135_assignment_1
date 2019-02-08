import numpy as np
from sklearn.model_selection import train_test_split

# X, y = np.random.rand(20, 28, 28), np.random.randint(0, 9, 20)
# X = X.reshape(20, 784)
X_f = np.loadtxt(open("/home/arthur/data/images.csv", "rb"), delimiter=",")
y_f = np.array(
    np.loadtxt(open("/home/arthur/data/labels.csv", "rb"), delimiter=","), dtype=int
)
X_train, X_test, y_train, y_test = train_test_split(
    X_f, y_f, test_size=0.2, shuffle=True, stratify=y_f, random_state=420
)

X = X_train[:15]
y = y_train[:15]
size1 = 500
size2 = 800
output_size = 10
learning_rate = 1e-7

input_size = X.shape[1]
W1 = np.random.normal(size=(input_size, size1))
b1 = 0.00 * np.ones((size1))
W2 = np.random.normal(size=(size1, size2))
b2 = 0.00 * np.ones((size2))
W3 = np.random.normal(size=(size2, output_size))
b3 = 0.00 * np.ones((output_size))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=0))
    return exps / np.sum(exps, axis=0)


def cross_entropy(y_true, y_pred):
    likelyhood = y_pred[range(len(y_true)), y_true]
    log_likelyhood = -np.log(likelyhood + 1e-15)
    return np.sum(log_likelyhood) / y_true.shape[0]


def score(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=0)
    good = 0
    for i, lab in enumerate(y_pred):
        good += int(lab == y_true[i])
    return good / len(y_true)


def onehot(y, n):
    onehots = []
    for lab in y:
        onehots.append(np.array([1 if i == lab else 0 for i in range(n)]))
    return np.asarray(onehots)


h1 = np.dot(X, W1) + b1
h2 = np.dot(h1, W2) + b2
out = np.dot(h2, W3) + b3
p = softmax(out)
loss = cross_entropy(y, p)
print(loss, score(p, y))

doa = p - onehot(y, out.shape[1])

dW3 = np.dot(h2.T, doa) / X.shape[0]
db3 = np.mean(doa, axis=0)

dh2 = np.dot(doa, W3.T)
dW2 = np.dot(h1.T, dh2) / X.shape[0]
db2 = np.mean(dh2, axis=0)

dh1 = np.dot(dh2, W2.T)
dW1 = np.dot(X.T, dh1) / X.shape[0]
db1 = np.mean(dh1, axis=0)

W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
W3 -= learning_rate * dW3
b3 -= learning_rate * db3
