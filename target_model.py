import tensorflow as tf
import numpy as np
import keras


class FCNetwork():

    def __init__(self):
        self.number_of_train_data = 5000
        self.number_of_test_data = 1000
        self.resize_width = 28
        self.resize_height = 28

    def load_data(self):
        (train_datas, train_labels), (test_datas,
                                      test_labels) = keras.datasets.mnist.load_data()
        train_labels = train_labels[:self.number_of_train_data]
        test_labels = test_labels[:self.number_of_test_data]
        train_datas = train_datas[:self.number_of_train_data].reshape(
            -1, self.resize_width * self.resize_height) / 255.0
        test_datas = test_datas[:self.number_of_test_data].reshape(
            -1, self.resize_width * self.resize_height) / 255.0

        # Standardize training samples
        mean_px = train_datas.mean().astype(np.float32)
        std_px = train_datas.std().astype(np.float32)
        train_datas = (train_datas - mean_px) / std_px

        # Standardize test samples
        mean_px = test_datas.mean().astype(np.float32)
        std_px = test_datas.std().astype(np.float32)
        test_datas = (test_datas - mean_px) / std_px

        # One-hot encoding the labels
        train_labels = keras.utils.np_utils.to_categorical(train_labels)
        test_labels = keras.utils.np_utils.to_categorical(test_labels)
        return (train_datas, train_labels), (test_datas, test_labels)

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)

    def create_simple_FC_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation='relu',
                               input_shape=(self.resize_width * self.resize_height, )),
            keras.layers.Dense(16, activation='relu', input_shape=(64, )),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_datas, train_labels, epochs=20, batch_size=None, callbacks=None, verbose=False):
        model.fit(train_datas, train_labels, epochs=epochs,
                  batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        return model

    def evaluate_model(self, model, test_datas, test_labels):
        loss, acc = model.evaluate(test_datas, test_labels)
        print('loss', loss)
        print('accurancy', acc)

    def save_model(self, model, name_of_file, mode='normal'):
        prefix = ''
        file_name = prefix + name_of_file + '.h5'
        model.save(file_name)
    
    def create_trained_simply_FC_model(self, verbose=False):
        (train_datas, train_labels), _ = self.load_data()
        model = self.create_simple_FC_model()
        model = self.compile_model(model)
        model = self.train_model(model, train_datas, train_labels)
        return model

    def create_and_save_trained_simply_FC_model(self, name_of_file, verbose=False):
        (train_datas, train_labels), _ = self.load_data()
        model = self.create_simple_FC_model()
        model = self.compile_model(model)
        model = self.train_model(model, train_datas, train_labels)
        self.save_model(model, name_of_file)
