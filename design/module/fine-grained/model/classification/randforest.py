#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import joblib
# import sklearnex
# patch_sklearn()

class RandomForest:
    def __init__(self, batch_size, x_train, y_train, scaler):
        # batch_size, x_train, y_train, and scaler are there to follow the other models' format
        # Initialize model
        self.model = RandomForestClassifier()

    def train(self, x_train, y_train, save=True, scaler_path = None, model_path = None, retrain=False):
        # Tree models won't benefit from both scaling and retraining
        print("Training", "."*20)

        # Train the model
        # Random Forest are not meant to be refitted
        self.model = self.model.fit(x_train, y_train)
        if save:
            joblib.dump(self.model, model_path)

        print("Done Training", "."*20)

    def pred(self, x_test):
        return self.model.predict(x_test)