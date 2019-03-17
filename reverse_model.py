import tensorflow as tf
import numpy as np
import keras
import keras.backend as K

def loss_func(y_true, y_pred):
    return keras.backend.mean(abs(y_true-y_pred)/y_true)

class FCNetwork():

    def __init__(self):
        self.train_datas = None
        self.train_labels = None
        self.test_datas = None
        self.test_labels = None

    def load_data(self, train_datas, train_labels, test_datas, test_labels):
        self.train_datas = train_datas
        self.train_labels = train_labels
        self.test_datas = test_datas
        self.test_labels = test_labels

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)

    def create_simple_FC_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(1200, activation='relu',
                               input_shape=(784, )),
            keras.layers.Dense(51450, activation='softmax', input_shape=(1200, ))
        ])
        return model

    def compile_model(self, model):
        def custom_loss(y_true, y_pred):
            return K.mean(K.square(K.abs(y_true - y_pred)))

        model.compile(optimizer='adam',
                      loss=custom_loss,
                      metrics=[keras.metrics.kullback_leibler_divergence])
        return model

    def train_model(self, model, train_datas, train_labels, epochs=10, batch_size=None, callbacks=None, verbose=False):
        print(train_datas.shape, train_labels.shape)
        model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
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
        model = self.create_simple_FC_model()
        model = self.compile_model(model)
        model = self.train_model(model, self.train_datas, self.train_labels)
        return model

    def create_and_save_trained_simply_FC_model(self, name_of_file, verbose=False):
        model = self.create_simple_FC_model()
        model = self.compile_model(model)
        model = self.train_model(model, self.train_datas, self.train_labels)
        self.save_model(model, name_of_file)
