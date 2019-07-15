import torch 
import torch.nn as nn
import torchvision
import numpy as np

HARD_CODED_PREFIX = 'Reverser/'

class WhiteboxModelExtractor():
    
    def __init__(self):
        # If GPU resource is avaiable, use GPU. Otherwise, use CPU. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_single_whitebox_model_weights(self, index):
        path = HARD_CODED_PREFIX + 'whitebox_database/model'+str(index)+'.pt'
        print(self.device)
        print(path)
        return torch.load(path)

    def extract_single_whitebox_model_outputs(self, index):
        path = HARD_CODED_PREFIX + 'query_outputs_database/query_outputs'+str(index)+'.pt'
        return torch.load(path)

    def extract_single_whitebox_model_predictions(self, index):
        path = HARD_CODED_PREFIX + 'query_predictions_database/query_predictions'+str(index)+'.pt'
        return torch.load(path)

    def parse_single_whitebox_model_weights(self, model):
        weights = np.array([])
        for _, v in model.items():
            layer_weights = v.cpu().numpy().flatten()
            weights = np.concatenate((weights, layer_weights))
        return weights

    def parse_single_whitebox_model_predictions(self, predictions):
        pass

    def extract_whitebox_model_weights(self, num_of_model_extracted):
        weights_dataset = np.array([])
        for i in range(num_of_model_extracted):
            model = self.extract_single_whitebox_model_weights(i+1)
            weights = self.parse_single_whitebox_model_weights(model)
            weights_dataset = np.concatenate((weights_dataset, weights), axis=0)
            
        return weights_dataset.reshape((num_of_model_extracted, -1))

    def extract_whitebox_model_outputs(self, num_of_model_extracted):
        outputs_dataset = np.array([])
        num_of_query, num_of_classes = None, None
        for i in range(num_of_model_extracted):
            outputs = self.extract_single_whitebox_model_outputs(i+1).cpu().numpy()
            if i == 0:
                num_of_query, num_of_classes = outputs.shape
            outputs_dataset = np.concatenate((outputs_dataset, outputs.flatten()))

        return outputs_dataset.reshape((num_of_model_extracted, num_of_query, num_of_classes))

    def extract_whitebox_model_predictions(self, num_of_model_extracted):
        predictions_dataset = np.array([])
        num_of_query = None
        for i in range(num_of_model_extracted):
            predictions = self.extract_single_whitebox_model_predictions(i+1).cpu().numpy()
            if i == 0:
                num_of_query = predictions.shape[0]
            predictions_dataset = np.concatenate((predictions_dataset, predictions))

        return predictions_dataset.reshape((num_of_model_extracted, num_of_query))
