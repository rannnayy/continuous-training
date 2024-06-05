#!/usr/bin/env python3

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import joblib

scaler_dict = {
    'MinMax': MinMaxScaler(),
    'MaxAbs': MaxAbsScaler(),
    'Standard': StandardScaler(),
    'Robust': RobustScaler(),
}

class DecisionTree:
    def __init__(self, batch_size, x_train, y_train, scaler=None):
        self.model = DecisionTreeClassifier()
        
        self.scaler = scaler
        if scaler != None:
            self.norm = scaler_dict[scaler]
        else:
            self.norm = None

    def train(self, x_train, y_train, save=True, scaler_path=None, model_path=None, retrain=False):
        print("Training", "."*20)

        # Data normalization
        if self.scaler != None:
            self.norm.fit(x_train)
            x_train = self.norm.transform(x_train)

        # Train the model
        self.model = self.model.fit(x_train.values, y_train.values)

        # Save
        if save:
            joblib.dump(self.model, model_path)
            if self.scaler != None:
                joblib.dump(self.norm, scaler_path)

        print("Done Training", "."*20)

    def pred(self, x_test):
        if self.scaler != None:
            x_test = self.norm.transform(x_test)
        return self.model.predict(x_test.values)

    def pred_proba(self, x_test, y_test):
        if self.scaler != None:
            x_test = self.norm.transform(x_test)
        classes = self.model.classes_.tolist()
        y_classes_idx = [classes.index(y) for y in y_test.tolist()]
        probabilities = self.model.predict_proba(x_test.values).tolist()
        correct_probabilities = [x_probas[y_idx] for x_probas, y_idx in zip(probabilities, y_classes_idx)]

        return correct_probabilities