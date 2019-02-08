import numpy as np
from sklearn.model_selection import train_test_split


def softmax(x):
    exps = np.exp(x - np.amax(x, axis=0))
    return exps / np.sum(exps, axis=0)


def cross_entropy(y_true, y_pred):
    n = y_true.shape[0]
    likelyhood = y_pred[range(n), y_true]
    log_likelyhood = -np.log(likelyhood + 1e-15)
    return np.sum(log_likelyhood) / n


def onehot(y, n):
    m = y.shape[0]
    onehots = np.zeros((m, n), dtype=int)
    for i, lab in enumerate(y):
        onehots[i, lab] = 1
    return np.asarray(onehots)


def not_onehot(y):
    return np.argmax(y, axis=0)


class two_layer_MLP:
    def __init__(self, input_size, size1, size2, output_size):
        self.W1 = np.random.normal(size=(size1, input_size))
        self.W2 = np.random.normal(size=(size2, size1))
        self.W3 = np.random.normal(size=(output_size, size2))
        self.b1 = 0.01 * np.ones((size1, 1))
        self.b2 = 0.01 * np.ones((size2, 1))
        self.b3 = 0.01 * np.ones((output_size, 1))

    def fit(self, X, y, batch_size, n_epochs, learning_rate=1e-8):
        score = 0
        for i in range(n_epochs):
            for k in range(0, len(X), batch_size):
                X_batch, y_batch = X[k : k + batch_size], y[k : k + batch_size]
                loss = self.sgd_step(X_batch, y_batch, learning_rate)
                score += sum((not_onehot(self.p) == y_batch).astype(int))
                if k % 1000 == 1:
                    print(
                        f"epoch #{i+1}: {loss:.2f}, {100*score / (k + i * len(X)):.2f}",
                        end="\r",
                    )
            print()

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.p, axis=0)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.sum((y_pred == y).astype(int)) / len(y)

    def forward(self, X):
        h1 = np.dot(self.W1, X.T) + self.b1
        h2 = np.dot(self.W2, h1) + self.b2
        out = np.dot(self.W3, h2) + self.b3
        self.p = softmax(out)
        return h2, h1

    def backward(self, X, y, h1, h2):
        doa = self.p - onehot(y, self.p.shape[1]).T
        dW3 = np.dot(doa, h2.T)  # / X.shape[0]
        db3 = np.mean(doa, axis=1).reshape(-1, 1)

        dh2 = np.dot(self.W3.T, doa)
        dW2 = np.dot(dh2, h1.T)  # / X.shape[0]
        db2 = np.mean(dh2, axis=1).reshape(-1, 1)

        dh1 = np.dot(self.W2.T, dh2)
        dW1 = np.dot(dh1, X)  # / X.shape[0]
        db1 = np.mean(dh1, axis=1).reshape(-1, 1)

        return dW1, db1, dW2, db2, dW3, db3

    def n_params(self):
        return 2 * self.W1.shape[0] + 2 * self.W2.shape[0]

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
        h2, h1 = self.forward(X)
        self.loss = cross_entropy(y, self.p)
        return self.backward(X, y, h1, h2)


if __name__ == "__main__":
    X = np.loadtxt(open("/home/arthur/data/images.csv", "rb"), delimiter=",")
    y = np.array(
        np.loadtxt(open("/home/arthur/data/labels.csv", "rb"), delimiter=","), dtype=int
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=420
    )
    model = two_layer_MLP(X.shape[1], 1024, 2048, 10)
    model.fit(X_train, y_train, batch_size=10, n_epochs=10)
    print(model.evaluate(X_test, y_test))
