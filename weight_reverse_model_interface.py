import torch 
import torch.nn as nn
import torchvision
import numpy as np
import random

class WeightReverseModelInterface():
    
    def __init__(self):
        # If GPU resource is avaiable, use GPU. Otherwise, use CPU. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_of_train_samples = 0
        self.weights_loader = None
        self.outputs_loader = None
        self.predictions_dataset = None

        self.model = None
        self.loss_func = None
        self.optimizer = None

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

    def set_dataset_loader(self, weights_dataset, outputs_dataset, predictions_dataset, batch_size):
        self.num_of_weights_per_model = weights_dataset.shape[1]
        self.batch_size = batch_size
        self.weights_loader = torch.utils.data.DataLoader(dataset=weights_dataset, batch_size=self.batch_size)
        self.outputs_loader = torch.utils.data.DataLoader(dataset=outputs_dataset, batch_size=self.batch_size)
        self.predictions_loader = torch.utils.data.DataLoader(dataset=predictions_dataset, batch_size=self.batch_size) 
        self.num_of_train_samples = len(weights_dataset)

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
                loss, _, _ = self.loss_func.forward(predicted_weights, weights, predicted_weights, predictions)

                # Optimization (back-propogation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % self.num_of_print_interval == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_of_epochs, i+1, total_step, loss.item()))
                    '''
                    1. maybe should use VAE, which is suitable for abundant inputs and outputs training
                    2. maybe separate weights into several pieces at train them individual and combine them to ensure the final prediction accurancy
                    '''
                    # print(predicted_weights[0][25000], weights[0][25000])

    def train_with_experiment(self, experiment_number):
        total_step = self.num_of_train_samples/self.batch_size
        combined_losses, l1_losses, l2_losses = [], [], []

        # Randomly select weight values for experiment 2
        selected_neuron_indexes = []
        data_for_experiment_2 = []
        for _ in range(10):
            selected_neuron_indexes.append((random.randint(0, self.batch_size-1), random.randint(0, self.num_of_weights_per_model-1)))
            data_for_experiment_2.append([])
        data_for_experiment_2.append([])

        for epoch in range(self.num_of_epochs):
            for i, (weights, outputs, predictions) in enumerate(zip(self.weights_loader, self.outputs_loader, self.predictions_loader)):
                
                # Move tensors to the configured device
                outputs = outputs.reshape(-1, self.input_size).to(self.device)
                weights = weights.to(self.device)

                # Forwarding 
                predicted_weights = self.model.forward(outputs)
                loss, l1, l2 = self.loss_func.forward(predicted_weights, weights, predicted_weights, predictions)
                
                # collect data for experiment 1
                combined_losses.append(loss.item())
                l1_losses.append(l1.item())
                l2_losses.append(l2.item())

                # Optimization (back-propogation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % self.num_of_print_interval == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_of_epochs, i+1, total_step, loss.item()))
                    cum_percentage_err = 0
                    for i, (model_idx, weight_idx) in enumerate(selected_neuron_indexes):
                        tar_val = weights[model_idx][weight_idx].item()
                        pre_val = predicted_weights[model_idx][weight_idx].item()
                        abs_diff = abs(tar_val - pre_val)
                        abs_tar_val = abs(tar_val)
                        abs_percentage_err = abs_diff / abs_tar_val
                        cum_percentage_err += abs_percentage_err
                        data_for_experiment_2[i].append(abs_percentage_err)

                    avg_percentage_err = cum_percentage_err / len(selected_neuron_indexes)
                    data_for_experiment_2[len(data_for_experiment_2)-1].append(avg_percentage_err)

        print('Data of experiment', experiment_number, 'is successfully computed and returned')

        results = None
        if experiment_number == 1:
            results = (combined_losses, l1_losses, l2_losses)
        elif experiment_number == 2:
            results = data_for_experiment_2

        return results

    def test(self, generation_index):
        '''
        1. how to test, using the original loss functions, absolutate percentage error?
        '''
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (weights, outputs, _) in enumerate(zip(self.weights_loader, self.outputs_loader, self.predictions_loader)):
                continue
            for _, (images, labels) in enumerate(self.test_loader):
                images = images.reshape(-1, self.input_size).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs.data, 1)

                self.save_query_outputs(outputs, './query_outputs_database/query_outputs'+str(generation_index+1)+'.pt')
                self.save_query_predictions(predictions, './query_predictions_database/query_predictions'+str(generation_index+1)+'.pt')

                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                
            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

    def save_model(self, str):
        torch.save(self.model.state_dict(), str)

    def save_query_outputs(self, query_outputs, str):
        torch.save(query_outputs, str)

    def save_query_predictions(self, query_predictions, str):
        torch.save(query_predictions, str)
        

    