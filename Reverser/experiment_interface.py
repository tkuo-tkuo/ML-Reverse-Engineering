import torch 
import torchvision
import torchvision.transforms as transforms
import numpy as np

from .input_generation import WhiteboxModelGenerator
from .input_generation import WhiteboxModelExtractor

from .weight_reverse_models import FC_WeightModel, VAE_WeightModel, FC_Loss, VAE_Loss
from .weight_reverse_model_interface import WeightReverseModelInterface

class ExperimentInterface():

    def __init__(self, weightmodel_architecture, num_of_model_extracted_for_training, num_of_model_extracted_for_testing, batch_size, num_of_epochs, learning_rate, num_of_print_interval):
        # Set internal variables 
        self.weightmodel_architecture = weightmodel_architecture
        self.num_of_model_extracted_for_training = num_of_model_extracted_for_training
        self.num_of_model_extracted_for_testing = num_of_model_extracted_for_testing

        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.num_of_print_interval = num_of_print_interval

        # Instanciate needed classes (whitebox extractor and weight_reverse_model interface)
        self.whitebox_extractor = WhiteboxModelExtractor()
        self.weight_reverse_model_interface = WeightReverseModelInterface(self.weightmodel_architecture) 
        self._init_weight_reverse_model_interface(self.weightmodel_architecture, learning_rate)

        # Set dataset for weight_reverse_model training and testing
        self.num_of_model_extracted = num_of_model_extracted_for_training + num_of_model_extracted_for_testing
        print('Extracting weights dataset...')
        weights_dataset = self.whitebox_extractor.extract_whitebox_model_weights(
            self.num_of_model_extracted)
        print('Extracting outputs dataset...')
        outputs_dataset = self.whitebox_extractor.extract_whitebox_model_outputs(
            self.num_of_model_extracted)
        print('Extracting predictions dataset...')
        predictions_dataset = self.whitebox_extractor.extract_whitebox_model_predictions(
            self.num_of_model_extracted)

        self._set_weightmodel_train_dataset(weights_dataset[:num_of_model_extracted_for_training], outputs_dataset[:num_of_model_extracted_for_training], predictions_dataset[:num_of_model_extracted_for_training], self.batch_size)
        self._set_weightmodel_test_dataset(weights_dataset[num_of_model_extracted_for_training:], outputs_dataset[num_of_model_extracted_for_training:], predictions_dataset[num_of_model_extracted_for_training:])

        # Set hyperparameters for weight_reverse_model 
        self._set_weightmodel_hyperparameters(num_of_epochs=self.num_of_epochs, num_of_print_interval=self.num_of_print_interval)

    def _init_weight_reverse_model_interface(self, architecture, lr):
        '''
        Set weight model architecture and its hyper parameters 
        '''
        if architecture == 'FC':
            input_size, hidden_size_1, hidden_size_2, output_size = 100000, 50, 50, 50890
            model = FC_WeightModel(input_size, hidden_size_1, hidden_size_2, output_size)
            loss = FC_Loss()
        elif architecture == 'VAE':
            model = VAE_WeightModel()
            loss = VAE_Loss()
        else: 
            raise ValueError(architecture, 'is not a valid architecture indication')

        # Set architecture and loss function for weight_reverse_model
        self.weight_reverse_model_interface.set_model(model)
        self.weight_reverse_model_interface.set_loss_func(loss)

        # Set optimizer of weight reverse model 
        self.weight_reverse_model_interface.set_optimizer(torch.optim.Adam(model.parameters(), lr=lr))

    def _set_weightmodel_train_dataset(self, weights_dataset, outputs_dataset, predictions_dataset, batch_size):
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        self.weight_reverse_model_interface.set_train_dataset_loader(weights_dataset, outputs_dataset, predictions_dataset, batch_size)

    def _set_weightmodel_test_dataset(self, weights_dataset, outputs_dataset, predictions_dataset):
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        self.weight_reverse_model_interface.set_test_dataset_loader(weights_dataset, outputs_dataset, predictions_dataset)

    def _set_weightmodel_hyperparameters(self, num_of_epochs=1, num_of_print_interval=1):
        input_size = 100000
        self.weight_reverse_model_interface.set_hyperparameters(num_of_epochs=num_of_epochs, num_of_print_interval=num_of_print_interval, input_size=input_size)

    def train_weightmodel(self):
        self.weight_reverse_model_interface.train()

    def test_weightmodel(self):
        self.weight_reverse_model_interface.test()

    def verify_weightmodel_reverse_effectiveness(self):
        self.weight_reverse_model_interface.verify()

    def generate_substitute_model(self):
        self.weight_reverse_model_interface.generate_substitute_model()
