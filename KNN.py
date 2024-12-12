from typing import List, Dict, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def train_test_split(
    X: np.ndarray, 
    y: np.ndarray,
    train_size: float, 
    random_state: int
):
    """ Randomizes and then splits the data into train and test sets. """
    rng = np.random.RandomState(random_state)
    random_idx = rng.permutation(len(X))
    X = X[random_idx]
    y = y[random_idx]
    
    split_idx = int(len(X) * train_size)
    X_trn, X_tst, y_trn, y_tst = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    return X_trn, X_tst, y_trn, y_tst

def euclidean_distance(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """ Compute the Euclidean distance between a row vector and a matrix. """
    assert len(Y.shape) == 2, f"Y is a 1D vector, expected 2D row vector or matrix"
    sub = Y - x
    squared_sub = sub ** 2
    sum = np.sum(squared_sub, axis=1)
    distance = np.sqrt(sum)    
    return distance

def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Computes the accuracy between two 1D vectors """
    y = np.array(y)
    y_hat = np.array(y_hat)
    y =  y.reshape(-1,)
    y_hat = y_hat.reshape(-1,)

    are_same = (y == y_hat)
    total_correct = np.sum(are_same)
    total_samples = len(y)
    
    score = total_correct / total_samples
    return score

class KNearestNeighbors():
    """ KNN algorithm class """
    def __init__(self, k: int, distance_measure: Callable):
        """
            Args:
                k: Number of nearest neighbors
                distance_measure: A function that computes a distance measure
        """
        self.k = k
        self.distance_measure = distance_measure
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Store the training data for comparison """
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Perform KNN using stored training data """
        y_hats: list = []
        for x_test in X:
            distances = self.distance_measure(x_test, self.X)
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y[nearest_neighbors]
            nearest_labels = nearest_labels.flatten()
            

            label_count = np.bincount(nearest_labels)
            prediction = np.argmax(label_count)
            y_hats.append(prediction)
            
        return np.array(y_hats)


df = pd.read_csv('dataset.csv')


features = ['danceability', 'energy', 'tempo', 'instrumentalness', 'valence']
X = df[features].values  
y = df['track_genre'].values


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_trn, X_tst, y_trn, y_tst = train_test_split(X, y_encoded, train_size=0.8, random_state=42)


knn = KNearestNeighbors(k=3, distance_measure=euclidean_distance)
knn.fit(X_trn, y_trn)


y_hat = knn.predict(X_tst)
y_hat_decoded = label_encoder.inverse_transform(y_hat)


print(f"Predictions: {y_hat_decoded}")
test_acc = accuracy(y=y_tst, y_hat=y_hat)
print(f"Test accuracy: {test_acc:.2f}")