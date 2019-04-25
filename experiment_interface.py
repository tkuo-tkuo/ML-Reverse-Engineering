import torch 
import torch.nn as nn
import torchvision
import numpy as np

import whitebox_model_generator
import whitebox_model_extractor

import weight_model_interface

class ExperimentInterface():

    def __init__(self):
        self.whitebox_generator = whitebox_model_generator.WhiteboxModelGenerator()
        self.whitebox_extractor = whitebox_model_extractor.WhiteboxModelExtractor()
        self.init_whitebox_generator()

        self.weight_model_interface = weight_model_interface.WeightModelInterface() 
        self.init_weight_model_interface()

    def init_whitebox_generator(self):
        # Set dataset loader, including train loader and test loader 
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)
        self.whitebox_generator.set_dataset_loader(train_loader, test_loader)

        # Set white-box architecture and its hyper parameters 
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.layer2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                output = self.layer1(x)
                output = self.relu(output)
                output = self.layer2(output)
                return output 

        input_size, hidden_size, output_size = 80000, 60000, 50890
        model = NeuralNet(input_size, hidden_size, output_size)
        self.whitebox_generator.set_model(model)

        # Set loss function and optimizer of white-box model 
        learning_rate = 0.001
        self.whitebox_generator.set_loss_func(nn.MSELoss())
        self.whitebox_generator.set_optimizer(torch.optim.Adam(model.parameters(), lr=learning_rate))

        # Set hyperparameters for the generation (training) proces
        self.whitebox_generator.set_generation_hyperparameters(num_of_epochs=1, num_of_print_interval=100, input_size=784)

    def init_weight_model_interface(self):
        # Set weight model architecture and its hyper parameters 
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

        input_size, hidden_size_1, hidden_size_2, output_size = 100000, 784, 64, 1
        model = NeuralNet(input_size, hidden_size_1, hidden_size_2, output_size)
        self.weight_model_interface.set_model(model)

        # Set loss function and optimizer of white-box model 
        learning_rate = 0.001
        self.weight_model_interface.set_loss_func(nn.CrossEntropyLoss())
        self.weight_model_interface.set_optimizer(torch.optim.Adam(model.parameters(), lr=learning_rate))

        # Set hyperparameters for the generation (training) proces
        num_of_epochs = 1
        num_of_print_interval = 1
        self.weight_model_interface.set_hyperparameters(num_of_epochs=num_of_epochs, num_of_print_interval=num_of_print_interval, input_size=input_size)

    def set_weightmodel_train_dataset(self, weights_dataset, outputs_dataset, predictions_dataset, num_of_train_samples):
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        weights_loader = torch.utils.data.DataLoader(dataset=weights_dataset)
        outputs_loader = torch.utils.data.DataLoader(dataset=outputs_dataset)
        predicts_loader = torch.utils.data.DataLoader(dataset=predictions_dataset) 
        self.weight_model_interface.set_dataset_loader(weights_loader, outputs_loader, predicts_loader, num_of_train_samples)

    def train_weightmodel(self):
        self.weight_model_interface.train()

    def generate_whitebox_model(self, num_of_model_generated):
        self.whitebox_generator.generate(num_of_model_generated)

    def extract_whitebox_model_weights(self, num_of_model_extracted):
        return self.whitebox_extractor.extract_whitebox_model_weights(num_of_model_extracted)

    def extract_whitebox_model_outputs(self, num_of_model_extracted):
        return self.whitebox_extractor.extract_whitebox_model_outputs(num_of_model_extracted)

    def extract_whitebox_model_predictions(self, num_of_model_extracted):
        return self.whitebox_extractor.extract_whitebox_model_predictions(num_of_model_extracted)


        

'''
functionality to generate trained whitebox (completed)
'''
interface = ExperimentInterface()
num_of_model_generated = 3
# interface.generate_whitebox_model(num_of_model_generated)

'''
functionality to extract data from trained whitebox (completed)
'''
num_of_model_extracted = 3
weights_dataset = interface.extract_whitebox_model_weights(num_of_model_extracted)
print(weights_dataset.shape)

outputs_dataset = interface.extract_whitebox_model_outputs(num_of_model_extracted)
print(outputs_dataset.shape)

predictions_dataset = interface.extract_whitebox_model_predictions(num_of_model_extracted)
print(predictions_dataset.shape)

'''
functionality to train the weight model
1. essemble a runable weight model 
'''
interface.set_weightmodel_train_dataset(weights_dataset, outputs_dataset, predictions_dataset, num_of_model_extracted)
# go on, man! (working......)
interface.train_weightmodel()
'''
problem encounter: how to train with dataset which datas and labels are separate, even in 3 individual dataset. 
-> merge these 3 datasets /(or) write some codes to deal with it?
'''



'''
functionality to generate predictions based on outputs
1. create an utils.py
'''
# from outputs (numpy) to predictions
# np_predictions_based_on_outputs = np.argmax(np_outputs, axis=1)
# print(np_predictions == np_predictions_based_on_outputs)



'''
functions are requried:
weights comparision
outputs comparision
'''