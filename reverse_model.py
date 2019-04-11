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
            keras.layers.Dense(360, activation='relu',
                               input_shape=(784, )),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu', input_shape=(360, )),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation='relu', input_shape=(64, )),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(10, activation=None, input_shape=(16, ))
        ])
        # model = keras.models.Sequential([
        #     keras.layers.Dense(1024, activation=keras.activations.relu,
        #                        input_shape=(784, )),
        #     keras.layers.Dropout(0.1),
        #     keras.layers.Dense(1024, activation=keras.activations.relu, input_shape=(1024, )),
        #     keras.layers.Dropout(0.1),
        #     keras.layers.Dense(2048, activation=keras.activations.relu, input_shape=(1024, )),            
        #     keras.layers.Dropout(0.1),
        #     keras.layers.Dense(51450, activation=None, input_shape=(2048, ))
        # ])
        return model

    def compile_model(self, model):
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
        model.compile(optimizer=optimizer,
                      loss=keras.losses.mean_squared_error,
                      metrics=[keras.metrics.mean_absolute_percentage_error])
        return model

    def train_model(self, model, train_datas, train_labels, epochs=1000, batch_size=5, callbacks=None, verbose=False):
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
