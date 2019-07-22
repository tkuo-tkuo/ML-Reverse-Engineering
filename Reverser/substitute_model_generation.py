import numpy as np

import torch
import torch.nn as nn
import torchvision

from .predictions_similarity_estimator import PredictionsSimilarityEstimator
from .input_generation import WhiteboxModelExtractor

class SubstituteModelGenerator():

    def __init__(self):
        self.verifier = PredictionsSimilarityEstimator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.whitebox_extractor = WhiteboxModelExtractor()

    def generate_initial_training_set(self, num_of_inital_training_set):
        initial_training_set = None
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=num_of_inital_training_set, shuffle=False)
        for i, (inputs, labels) in enumerate(test_loader):
            initial_training_set = inputs.reshape(-1, 784).to(self.device)
            break 
        return initial_training_set

    def generate_target_black_box(self):
        weights_dataset = self.whitebox_extractor.extract_whitebox_model_weights(1)
        outputs_dataset = self.whitebox_extractor.extract_whitebox_model_outputs(1)
        predictions_dataset = self.whitebox_extractor.extract_whitebox_model_predictions(1)
        weights_dataset = np.float32(weights_dataset)
        outputs_dataset = np.float32(outputs_dataset)
        predictions_dataset = np.float32(predictions_dataset)
        weights_loader = torch.utils.data.DataLoader(dataset=weights_dataset, batch_size=1)
        outputs_loader = torch.utils.data.DataLoader(dataset=outputs_dataset, batch_size=1)
        predictions_loader = torch.utils.data.DataLoader(dataset=predictions_dataset, batch_size=1)


        self.verifier = PredictionsSimilarityEstimator()
        with torch.no_grad():
            for i, (weights, outputs, predictions) in enumerate(zip(weights_loader, outputs_loader, predictions_loader)):

                weights = weights.to(self.device)
                weights_of_black_box = weights[0]
                self.verifier.set_on_a_black_box_model(weights_of_black_box)
                break

        return self.verifier

    def generate_f_prime_model(self):
        class WhiteboxNeuralNet(nn.Module):
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

        input_size, hidden_size, output_size = 784, 64, 10
        f_prime_model = WhiteboxNeuralNet(input_size, hidden_size, output_size)   
        f_prime_model = f_prime_model.to(self.device) 
        return f_prime_model

    def evaluate(self, f_prime, f):
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2000, shuffle=False)
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.reshape(-1, 784).to(self.device)
            _, f_prime_predictions = torch.max(f_prime.forward(inputs).data, 1)
            f_predictions = f.query_black_box_model(inputs)

            match = (f_prime_predictions == f_predictions).sum().item()
            print('Prediction mismatch ratio:', 1 - match/2000, '/1')
            break

        f_prime_weights = self.whitebox_extractor.parse_single_whitebox_model_weights(f_prime.state_dict())
        f_weights = self.whitebox_extractor.parse_single_whitebox_model_weights(f.forward_model.state_dict())
        mean_APE = np.mean(np.abs((f_weights - f_prime_weights) / f_weights)) * 100
        print('Mean absolute percentage error:', mean_APE, '%')

        print('l2 norm:', np.linalg.norm(f_prime_weights-f_weights))
        print()

    def generate_substitute_model(self):
        num_of_epochs_for_reversing = 5000  # ρ
        num_of_inital_training_set = 2000

        # Setup initial training set S0
        S = self.generate_initial_training_set(num_of_inital_training_set) 

        # Setup a target black-box model f with architecture F
        f = self.generate_target_black_box()

        # Setup a model f' to approxiate f with architecture F (WORKING)
        f_prime_model = self.generate_f_prime_model()
        
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim .Adam(f_prime_model.parameters(), lr=1e-4)
        
        for i in range(num_of_epochs_for_reversing):
            # Label the substitute training set S_i and get the relation D_i
            O_S = f.query_black_box_model(S)

            # Train f' with D_i
            f_prime_outpus = f_prime_model.forward(S)
            loss = loss_func(f_prime_outpus, O_S)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Jacobian-based dataset augmentation to get S_i+1
            # PENDING


            # Evaluate (DEBUG)
            if (i+1)%500 == 0:
                self.evaluate(f_prime_model, f)

        # Obtain f' & Evaluate f'
        # PENDING
        pass 


