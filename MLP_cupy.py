from time import time
import cupy as cp
import numpy as np
from sklearn.model_selection import train_test_split


def softmax(x):
    exps = cp.exp(x - cp.max(x, axis=0))
    return exps / cp.sum(exps, axis=0)


def cross_entropy(y_true, y_pred):
    n = y_true.shape[0]
    likelyhood = y_pred[cp.arange(n), y_true]
    log_likelyhood = -cp.log(likelyhood + 1e-15)
    return cp.sum(log_likelyhood) / n


def onehot(y, n):
    m = y.shape[0]
    onehots = cp.zeros((m, n), dtype=int)
    for i, lab in enumerate(y):
        onehots[i, lab] = 1
    return cp.asarray(onehots)


def not_onehot(y):
    return cp.argmax(y, axis=0)


def dropout(X, p):
    shape = X.shape
    drop_idx = np.random.choice(np.arange(X.size), replace=False, size=int(X.size * p))
    X = X.flatten()
    X[drop_idx] = 0
    return X.reshape(shape)


def dReLU(X):
    X[X < 0] = 0
    X[X > 0] = 1
    return X


def ReLU(X):
    X[X < 0] = 0
    return X


class two_layer_MLP:
    def __init__(self, input_size, size1, size2, output_size):
        self.W1 = cp.random.uniform(size=(size1, input_size)) * cp.sqrt(
            6 / (input_size + size1)
        )
        self.W2 = cp.random.uniform(size=(size2, size1)) * cp.sqrt(6 / (size2 + size1))
        self.W3 = cp.random.uniform(size=(output_size, size2)) * cp.sqrt(
            6 / (output_size + size2)
        )
        self.b1 = 0.01 * cp.ones((size1, 1))
        self.b2 = 0.01 * cp.ones((size2, 1))
        self.b3 = 0.01 * cp.ones((output_size, 1))
        self.n_params = (
            size1 * (input_size + 1) + size2 * (size1 + 1) + output_size * (size2 + 1)
        )

    def fit(self, X, y, batch_size, n_epochs, learning_rate=1e-3):
        score = 0
        N = len(X)
        for i in range(n_epochs):
            for k in range(0, N, batch_size):
                X_batch, y_batch = X[k : k + batch_size], y[k : k + batch_size]
                loss = float(self.sgd_step(X_batch, y_batch, learning_rate))
                score += sum((not_onehot(self.p) == y_batch).astype(int))
                if k % 1000 == 0 and k != 0:
                    current_score = 100 * float(score / (k + i * N))
                    print(
                        f"epoch #{i+1} - loss: {loss:.2f}, training_accuracy: {current_score:.2f}",
                        end="\r",
                    )
            learning_rate -= learning_rate / 20
            print()

    def predict(self, X):
        self.forward(X)
        return cp.argmax(self.p, axis=0)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return cp.sum((y_pred == y).astype(int)) / len(y)

    def forward(self, X):
        h1 = cp.dot(self.W1, X.T) + self.b1
        a1 = ReLU(h1)
        d1 = dropout(a1, 0.2)
        h2 = cp.dot(self.W2, d1) + self.b2
        a2 = ReLU(h2)
        d2 = dropout(a2, 0.2)
        out = cp.dot(self.W3, d2) + self.b3
        self.p = softmax(out)
        return h1, a1, h2, a2

    def backward(self, X, y, h1, a1, h2, a2):
        doa = self.p - onehot(y, self.p.shape[0]).T
        dW3 = cp.dot(doa, a2.T) / X.shape[0]
        db3 = cp.mean(doa, axis=1).reshape(-1, 1)

        da2 = cp.dot(self.W3.T, doa)
        dh2 = da2 * dReLU(a2)
        dW2 = cp.dot(dh2, a1.T) / X.shape[0]
        db2 = cp.mean(da2, axis=1).reshape(-1, 1)

        da1 = cp.dot(self.W2.T, da2)
        dh1 = da1 * dReLU(a1)
        dW1 = cp.dot(dh1, X) / X.shape[0]
        db1 = cp.mean(da1, axis=1).reshape(-1, 1)

        return dW1, db1, dW2, db2, dW3, db3

    def sgd_step(self, X, y, learning_rate):
        dW1, db1, dW2, db2, dW3, db3 = self.compute_grad(X, y)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        return self.loss

    def compute_grad(self, X, y):
        h1, a1, h2, a2 = self.forward(X)
        self.loss = cross_entropy(y, self.p)
        return self.backward(X, y, h1, a1, h2, a2)


if __name__ == "__main__":
    X = cp.load("/home/arthur/data/MNIST/digits.npy") / 255
    X = X.astype(np.float32)
    y = cp.load("/home/arthur/data/MNIST/target.npy")
    index = cp.arange(len(X))
    cp.random.RandomState(seed=0).shuffle(index)
    X = X[index]
    y = y[index]
    split_id = int(0.8 * len(X))
    X_train, X_test = X[:split_id], X[split_id:]
    y_train, y_test = y[:split_id], y[split_id:]

    model = two_layer_MLP(X.shape[1], 512, 512, 10)
    start = time()
    model.fit(X_train, y_train, batch_size=10, n_epochs=100, learning_rate=0.001)
    print("total time :", time() - start)
    print("Test accuracy:", model.evaluate(X_test, y_test))
