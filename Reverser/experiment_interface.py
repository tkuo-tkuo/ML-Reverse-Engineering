# from __future__ import absolute_import

import torch 
import torch.nn as nn
import torchvision
import numpy as np

from .training_whiteboxs import WhiteboxModelGenerator
from .training_whiteboxs import WhiteboxModelExtractor

from .weight_reverse_model_interface import WeightReverseModelInterface
from .customized_loss_func import CustomerizedLoss


class FC_WeightModel(nn.Module):
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

class VAE_WeightModel(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class ExperimentInterface():

    def __init__(self, weightmodel_architecture, num_of_model_extracted_for_training, num_of_model_extracted_for_testing, batch_size, num_of_epochs, num_of_print_interval):
        # Set internal variables 
        self.weightmodel_architecture = weightmodel_architecture

        self.num_of_model_extracted_for_training = num_of_model_extracted_for_training
        self.num_of_model_extracted_for_testing = num_of_model_extracted_for_testing

        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.num_of_print_interval = num_of_print_interval

        # Instanciate needed classes (whitebox extractor and weight_reverse_model interface)
        self.whitebox_extractor = WhiteboxModelExtractor()
        self.weight_reverse_model_interface = WeightReverseModelInterface() 
        self.init_weight_reverse_model_interface(self.weightmodel_architecture)

        # Set dataset for weight_reverse_model training and testing
        self.num_of_model_extracted = num_of_model_extracted_for_training + num_of_model_extracted_for_testing
        weights_dataset = self.whitebox_extractor.extract_whitebox_model_weights(
            self.num_of_model_extracted)
        outputs_dataset = self.whitebox_extractor.extract_whitebox_model_outputs(
            self.num_of_model_extracted)
        predictions_dataset = self.whitebox_extractor.extract_whitebox_model_predictions(
            self.num_of_model_extracted)

        self.set_weightmodel_train_dataset(weights_dataset[:num_of_model_extracted_for_training], outputs_dataset[:num_of_model_extracted_for_training], predictions_dataset[:num_of_model_extracted_for_training], self.batch_size)
        self.set_weightmodel_test_dataset(weights_dataset[num_of_model_extracted_for_training:], outputs_dataset[num_of_model_extracted_for_training:], predictions_dataset[num_of_model_extracted_for_training:])

        # Set hyperparameters for weight_reverse_model 
        self.set_weightmodel_hyperparameters(num_of_epochs=self.num_of_epochs, num_of_print_interval=self.num_of_print_interval)

    def init_weight_reverse_model_interface(self, architecture):
        '''
        Set weight model architecture and its hyper parameters 
        '''
        if architecture == 'FC':
            input_size, hidden_size_1, hidden_size_2, output_size = 100000, 50, 50, 50890
            model = FC_WeightModel(input_size, hidden_size_1, hidden_size_2, output_size)

            # Set loss function of weight reverse model
            self.weight_reverse_model_interface.set_loss_func(CustomerizedLoss())
        elif architecture == 'VAE':
            model = VAE_WeightModel()

            def loss_function(recon_x, x, mu, logvar):
                BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

                # see Appendix B from VAE paper:
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                return BCE + KLD
            self.weight_reverse_model_interface.set_loss_func(loss_function)
        else: 
            raise ValueError(architecture, 'is not a valid architecture indication')

        self.weight_reverse_model_interface.set_model(model)

        # Set optimizer of weight reverse model 
        learning_rate = 0.001
        self.weight_reverse_model_interface.set_optimizer(torch.optim.Adam(model.parameters(), lr=learning_rate))

    def set_weightmodel_train_dataset(self, weights_dataset, outputs_dataset, predictions_dataset, batch_size):
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        self.weight_reverse_model_interface.set_train_dataset_loader(weights_dataset, outputs_dataset, predictions_dataset, batch_size)

    def set_weightmodel_test_dataset(self, weights_dataset, outputs_dataset, predictions_dataset):
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        self.weight_reverse_model_interface.set_test_dataset_loader(weights_dataset, outputs_dataset, predictions_dataset)
 

    def set_weightmodel_hyperparameters(self, num_of_epochs=1, num_of_print_interval=1):
        input_size = 100000
        self.weight_reverse_model_interface.set_hyperparameters(num_of_epochs=num_of_epochs, num_of_print_interval=num_of_print_interval, input_size=input_size)

    def train_weightmodel(self):
        self.weight_reverse_model_interface.train()

    # WORKING
    def test_weightmodel(self):
        self.weight_reverse_model_interface.test()
