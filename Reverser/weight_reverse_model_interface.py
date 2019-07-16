import torch 
import torch.nn as nn
import torchvision
import numpy as np
import random

class WeightReverseModelInterface():
    
    def __init__(self):
        # If GPU resource is avaiable, use GPU. Otherwise, use CPU. 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        self.model = None
        self.loss_func = None
        self.optimizer = None

        # Training process parameters 
        self.num_of_train_samples = 0
        self.train_weights_loader = None
        self.train_outputs_loader = None
        self.train_predictions_dataset = None

        # Testing process parameters 
        self.num_of_test_samples = 0
        self.test_weights_loader = None
        self.test_outputs_loader = None
        self.test_predictions_loader = None 

        # Hyper parameters
        self.num_of_epochs = 1
        self.batch_size = 1
        self.num_of_print_interval = 1
        self.input_size = 100000
        self.num_of_weights_per_model = None

    def set_hyperparameters(self, num_of_epochs=1, num_of_print_interval=1, input_size=100000):
        self.num_of_epochs = num_of_epochs
        self.num_of_print_interval = num_of_print_interval
        self.input_size = input_size

    def set_train_dataset_loader(self, weights_dataset, outputs_dataset, predictions_dataset, batch_size):
        self.num_of_weights_per_model = weights_dataset.shape[1]
        self.batch_size = batch_size
        
        self.train_weights_loader = torch.utils.data.DataLoader(dataset=weights_dataset, batch_size=self.batch_size)
        self.train_outputs_loader = torch.utils.data.DataLoader(dataset=outputs_dataset, batch_size=self.batch_size)
        self.train_predictions_dataset = torch.utils.data.DataLoader(dataset=predictions_dataset, batch_size=self.batch_size) 
        
        self.num_of_train_samples = len(weights_dataset)

    def set_test_dataset_loader(self, weights_dataset, outputs_dataset, predictions_dataset):
        self.test_weights_loader = torch.utils.data.DataLoader(dataset=weights_dataset)
        self.test_outputs_loader = torch.utils.data.DataLoader(dataset=outputs_dataset)
        self.test_predictions_loader = torch.utils.data.DataLoader(dataset=predictions_dataset) 
        
        self.num_of_test_samples = len(weights_dataset)

    def set_model(self, model):
        model = model.to(self.device)
        self.model = model      

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self):
        total_step = self.num_of_train_samples/self.batch_size
        for epoch in range(self.num_of_epochs):
            for i, (weights, outputs, predictions) in enumerate(zip(self.weights_loader, self.outputs_loader, self.predictions_loader)):
                
                # Move tensors to the configured device
                outputs = outputs.reshape(-1, self.input_size).to(self.device)
                weights = weights.to(self.device)

                # Forwarding 
                predicted_weights = self.model.forward(outputs)
                # loss = self.loss_func.forward(predicted_weights, weights, weights, predictions)
                loss, l1, l2 = self.loss_func.forward(predicted_weights, weights, predicted_weights, predictions)
                print(loss, l1, l2)

                # Optimization (back-propogation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % self.num_of_print_interval == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_of_epochs, i+1, total_step, loss.item()))
                    # print(predicted_weights[0][25000], weights[0][25000])

    def test(self, generation_index):
        with torch.no_grad():
            pass 


    
