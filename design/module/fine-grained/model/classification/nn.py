#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import joblib
import numpy as np

scaler_dict = {
    'MinMax': MinMaxScaler(),
    'MaxAbs': MaxAbsScaler(),
    'Standard': StandardScaler(),
    'Robust': RobustScaler(),
}

BATCH_SIZE = 256

class NN:
    def __init__(self, batch_size, x_train, y_train, scaler):
        # Initialize Model
        self.dnn_model = keras.Sequential()
        if scaler == 'BatchNorm':
            normalizer = layers.Normalization(axis=-1)
            normalizer.adapt(np.array(x_train))
            self.dnn_model.add(normalizer)
        self.dnn_model.add(layers.Dense(128, activation='relu', input_dim=x_train.shape[1]))
        self.dnn_model.add(layers.Dense(16, activation='relu'))
        self.dnn_model.add(layers.Dense(1, activation='sigmoid'))
        self.dnn_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        
        # Initialize Normalizer
        self.batch_size = batch_size
        self.scaler = scaler
        if scaler != None:
            self.norm = MinMaxScaler()
        else:
            self.norm = None

    def dataframe_generator(self, df, y_train):
        for i in range(0, len(df), self.batch_size):
            yield df[i:i+self.batch_size], y_train[i:i+self.batch_size]

    def train(self, x_train, y_train, save=True, scaler_path=None, model_path=None, retrain=False):
        print("Training", "."*20)

        # Data normalization
        if self.scaler != None:
            self.norm.fit(x_train)
            x_train = self.norm.transform(x_train)
        
        x_train_tensor = tf.data.Dataset.from_generator(
            self.dataframe_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, x_train.shape[1]), dtype=tf.float16),
                tf.TensorSpec(shape=(None,), dtype=tf.bool)
            ),
            args=(x_train, y_train)
        )

        # Model Architecture
        if retrain:
            self.dnn_model = tf.keras.models.load_model(self.model_path)
        
        # Train the model
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.01)
        self.dnn_model.fit(
            x_train_tensor,
            batch_size=self.batch_size,
            verbose=0, epochs=20,
            callbacks=[callback]
        )

        if save:
            self.dnn_model.save(model_path)
            if self.scaler != None:
                joblib.dump(self.norm, scaler_path)

        print("Done Training", "."*20)

        # return self.norm, self.dnn_model

    def pred(self, x_test):
        if self.scaler != None:
            x_test = self.norm.transform(x_test)
        x_test_tensor = tf.convert_to_tensor(x_test)
        return (self.dnn_model.predict(x_test_tensor, verbose=0) > 0.5).flatten()