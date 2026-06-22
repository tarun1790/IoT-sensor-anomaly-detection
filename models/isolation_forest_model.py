from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import os

class IsolationForestWrapper:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        
    def fit(self, X):
        """
        X: numpy array of shape (n_samples, n_features)
        """
        self.model.fit(X)
        
    def predict_anomaly(self, X):
        """
        X: numpy array of shape (n_samples, n_features)
        Returns: 1 if anomaly, 0 if normal
        """
        preds = self.model.predict(X) # returns 1 for inliers, -1 for outliers
        return np.where(preds == -1, 1, 0)
        
    def score_samples(self, X):
        """
        Returns raw anomaly scores. Lower values mean more anomalous.
        """
        return self.model.score_samples(X)
        
    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)
