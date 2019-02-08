h1 = np.dot(W1, X.T) + b1
h2 = np.dot(W2, h1) + b2
out = np.dot(W3, h2) + b3
out = out.T
p = softmax(out)
loss = cross_entropy(y, p)
print(loss, score(p, y))

doa = out - onehot(y, out.shape[1])

dW3 = np.dot(h2, doa).T / X.shape[0]
db3 = np.mean(doa, axis=0).reshape(b3.shape)
dh2 = np.dot(doa, W3)
db2 = np.mean(dh2, axis=0).reshape(b2.shape)
dW2 = np.dot(h1, dh2).T / X.shape[0]
dh1 = np.dot(dh2, W2)
db1 = np.mean(dh1, axis=0).reshape(b1.shape)
dW1 = np.dot(dh1.T, X) / X.shape[0]

W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
W3 -= learning_rate * dW3
b3 -= learning_rate * db3
