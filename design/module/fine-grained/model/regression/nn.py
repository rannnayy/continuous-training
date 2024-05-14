#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import joblib

BATCH_SIZE = 256

class NN:
    def __init__(self, batch_size, num_feature, scaler_path, model_path):
        self.dnn_model = keras.Sequential()
        self.dnn_model.add(layers.Dense(128, activation='relu', input_dim=num_feature))
        self.dnn_model.add(layers.Dense(16, activation='relu'))
        self.dnn_model.add(layers.Dense(1, activation='linear'))
        self.dnn_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['mae'])

        self.norm = MinMaxScaler()
        self.batch_size = batch_size
        self.scaler_path = scaler_path
        self.model_path = model_path

    def dataframe_generator(self, df, y_train):
        for i in range(0, len(df), self.batch_size):
            yield df[i:i+self.batch_size], y_train[i:i+self.batch_size]

    def train(self, x_train, y_train, retrain=False):
        print("Training", "."*20)

        # Data normalization
        self.norm.fit(x_train)
        x_train_norm = self.norm.transform(x_train)
        x_train_tensor = tf.data.Dataset.from_generator(
            self.dataframe_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, x_train.shape[1]), dtype=tf.float16),
                tf.TensorSpec(shape=(None,), dtype=tf.bool)
            ),
            args=(x_train_norm, y_train, self.batch_size)
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

        self.dnn_model.save(self.model_path)
        joblib.dump(self.norm, self.scaler_path)

        print("Done Training", "."*20)

    def pred(self, x_test):
        x_test_norm = self.norm.transform(x_test)
        x_test_tensor = tf.convert_to_tensor(x_test_norm)
        return self.dnn_model.predict(x_test_tensor, verbose=0).flatten()