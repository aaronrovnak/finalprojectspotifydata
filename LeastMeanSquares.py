import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LeastMeanSquares:
    def __init__(self, alpha=0.01, batch_size=32, seed=None, epochs=100):
        np.random.seed(seed)
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.trn_error = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                predictions = np.dot(X_batch, self.weights) + self.bias
                errors = predictions - y_batch

                
                grad_w = (2 / len(X_batch)) * np.dot(X_batch.T, errors)
                grad_b = (2 / len(X_batch)) * np.sum(errors)
                
                self.weights -= self.alpha * grad_w
                self.bias -= self.alpha * grad_b

            
            epoch_predictions = np.dot(X, self.weights) + self.bias
            epoch_error = np.mean((y - epoch_predictions)**2)
            self.trn_error.append(epoch_error)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



def get_preprocessed_data():
    data = pd.read_csv('dataset.csv')
    features = ['danceability', 'energy', 'tempo', 'instrumentalness', 'valence']
    target = 'popularity'

    X = data[features].values
    y = data[target].values

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    split_idx = int(0.8 * len(X))
    X_trn, X_tst = X[:split_idx], X[split_idx:]
    y_trn, y_tst = y[:split_idx], y[split_idx:]

    return X_trn, y_trn, X_tst, y_tst

X_trn, y_trn, X_tst, y_tst = get_preprocessed_data()

lms = LeastMeanSquares(alpha=0.0001, batch_size=128, seed=42, epochs=50)
lms.fit(X_trn, y_trn)


trn_preds = lms.predict(X_trn)
trn_mse = np.mean((y_trn - trn_preds)**2)
trn_rmse = np.sqrt(trn_mse)

test_preds = lms.predict(X_tst)
test_mse = np.mean((y_tst - test_preds)**2)
test_rmse = np.sqrt(test_mse)



print(f"Training RMSE: {trn_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
plt.plot(lms.trn_error, label='Train Error')
plt.title("LMS Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()
