# from __future__ import absolute_import

import torch 
import torch.nn as nn
import torchvision
import numpy as np

from .training_whiteboxs import WhiteboxModelGenerator
from .training_whiteboxs import WhiteboxModelExtractor

from .weight_reverse_model_interface import WeightReverseModelInterface
from .customized_loss_func import CustomerizedLoss

class ExperimentInterface():

    def __init__(self, num_of_model_extracted, batch_size, num_of_epochs, num_of_print_interval):
        self.num_of_model_extracted = num_of_model_extracted
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.num_of_print_interval = num_of_print_interval

        self.whitebox_extractor = WhiteboxModelExtractor()
        self.weight_reverse_model_interface = WeightReverseModelInterface() 
        self.init_weight_reverse_model_interface()

        self.num_of_model_extracted = num_of_model_extracted
        self.set_weightmodel_train_dataset(
            self.whitebox_extractor.extract_whitebox_model_weights(self.num_of_model_extracted), 
            self.whitebox_extractor.extract_whitebox_model_outputs(self.num_of_model_extracted),
            self.whitebox_extractor.extract_whitebox_model_predictions(self.num_of_model_extracted),
            self.batch_size)

        self.set_weightmodel_hyperparameters(num_of_epochs=self.num_of_epochs, num_of_print_interval=self.num_of_print_interval)

    def init_weight_reverse_model_interface(self):
        '''
        Set weight model architecture and its hyper parameters 
        '''
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
                super().__init__()
                self.layer1 = nn.Linear(input_size, hidden_size_1)
                self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
                self.layer3 = nn.Linear(hidden_size_2, output_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                output = self.layer1(x)
                output = self.relu(output)
                output = self.layer2(output)
                output = self.relu(output)
                output = self.layer3(output)
                return output 

        input_size, hidden_size_1, hidden_size_2, output_size = 100000, 2, 2, 50890
        model = NeuralNet(input_size, hidden_size_1, hidden_size_2, output_size)
        self.weight_reverse_model_interface.set_model(model)

        # Set loss function of weight reverse model 
        self.weight_reverse_model_interface.set_loss_func(CustomerizedLoss())

        # Set optimizer of weight reverse model 
        learning_rate = 0.001
        self.weight_reverse_model_interface.set_optimizer(torch.optim.Adam(model.parameters(), lr=learning_rate))

    def set_weightmodel_train_dataset(self, weights_dataset, outputs_dataset, predictions_dataset, batch_size):
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        self.weight_reverse_model_interface.set_dataset_loader(weights_dataset, outputs_dataset, predictions_dataset, batch_size)

    def set_weightmodel_hyperparameters(self, num_of_epochs=1, num_of_print_interval=1):
        input_size = 100000
        self.weight_reverse_model_interface.set_hyperparameters(num_of_epochs=num_of_epochs, num_of_print_interval=num_of_print_interval, input_size=input_size)

    def train_weightmodel(self):
        self.weight_reverse_model_interface.train()

    # PENDING
    def test_weightmodel(self):
        pass
