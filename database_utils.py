# import all the basic settings or essentail files
import target_model, feature_extraction_utils
import numpy as np
import pickle

class DatabaseGenerationUtils():

    def __init__(self):
        self.network = target_model.FCNetwork()
        self.feature_extraction_utils = feature_extraction_utils.FeatureExtrationUtils()

    def convert_weights_to_1D_array(self, weights):
        flatten_weights = np.array([])
        for weight in weights:
            flatten_weight = weight.flatten()
            flatten_weights = np.concatenate((flatten_weights, flatten_weight))
        return flatten_weights

    def generate_database_samples(self, number_of_samples_generated, average_degree, weights_database_name, attention_database_name):

        sensitivity_image_list = []
        weights_list = []
        _, (test_datas, test_labels) = self.network.load_data()

        for index in range(number_of_samples_generated):
            model = self.network.create_trained_simply_FC_model()
            sensitivity_image = np.zeros_like(test_datas[0])
            for i in range(average_degree):
                image_sample_data, image_sample_label = test_datas[i], test_labels[i]
                sensitivity_image_i = self.feature_extraction_utils.extract_attention(model, image_sample_data, image_sample_label)
                sensitivity_image = sensitivity_image + sensitivity_image_i
                print('processing sample', index+1, ',', str(i+1), '/', average_degree)

            sensitivity_image /= average_degree
            weights = np.array(model.get_weights())
            flatten_weights = self.convert_weights_to_1D_array(weights)
            weights_list.append(flatten_weights.tolist())
            sensitivity_image_list.append(sensitivity_image)
            print('Sample', index+1, 'is successfully created and stored')

        with open(attention_database_name, 'wb') as fp:
            pickle.dump(sensitivity_image_list, fp)
        with open(weights_database_name, 'wb') as fp:
            pickle.dump(weights_list, fp)


    def load_database_samples(self, weights_database_name, attention_database_name):
        weights_database = None
        attention_database = None
        with open(weights_database_name, 'rb') as fp:
            weights_database = np.array(pickle.load(fp))

        with open(attention_database_name, 'rb') as fp:
            attention_database = np.array(pickle.load(fp))

        return weights_database, attention_database
