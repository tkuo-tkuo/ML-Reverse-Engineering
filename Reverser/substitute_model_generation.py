import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable 

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
        num_of_test_samples = 5000
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=num_of_test_samples, shuffle=False)
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.reshape(-1, 784).to(self.device)
            _, f_prime_predictions = torch.max(f_prime.forward(inputs).data, 1)
            f_predictions = f.query_black_box_model(inputs)

            match = (f_prime_predictions == f_predictions).sum().item()
            print('Prediction mismatch ratio:', 1 - match/num_of_test_samples, '/1')
            break

        f_prime_weights = self.whitebox_extractor.parse_single_whitebox_model_weights(f_prime.state_dict())
        f_weights = self.whitebox_extractor.parse_single_whitebox_model_weights(f.forward_model.state_dict())
        mean_APE = np.mean(np.abs((f_weights - f_prime_weights) / f_weights)) * 100
        print('Mean absolute percentage error:', mean_APE, '%')

        print('l2 norm:', np.linalg.norm(f_prime_weights-f_weights))
        print()

    def to_var(self, x, requires_grad=False, volatile=False):
        '''
        Varialbe type that automatically choose cpu or cuda
        '''
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad, volatile=volatile)

    def jacobian(self, model, x, nb_classes=10):
        '''
        This function will return a list of PyTorch gradients
        '''
        list_derivatives = []
        x_var = self.to_var(x, requires_grad=True)

        # derivatives for each class
        for class_ind in range(nb_classes):
            score = model(x_var)
            # score = score[:, class_ind]
            score = score[class_ind]
            score.backward()
            list_derivatives.append(x_var.grad.data.cpu().numpy())
            x_var.grad.data.zero_()

        return list_derivatives


    def jacobian_augmentation(self, model, X_sub_prev, Y_sub, lmbda=0.5):
        '''
        Create new numpy array for adversary training data
        with twice as many components on the first dimension.
        '''
        X_sub = np.vstack([X_sub_prev.cpu(), X_sub_prev.cpu()])

        # For each input in the previous' substitute training iteration
        for ind, x in enumerate(X_sub_prev):
            grads = self.jacobian(model, x)
            # Select gradient corresponding to the label predicted by the oracle
            grad = grads[Y_sub[ind]]

            # Compute sign matrix
            grad_val = np.sign(grad)
            # Create new synthetic point in adversary substitute training set
            X_sub[len(X_sub_prev)+ind] = X_sub[ind] - lmbda * grad_val  

        # Return augmented training data (needs to be labeled afterwards)
        return X_sub

    def generate_substitute_model(self):
        num_of_epochs_for_reversing = 5000  # ρ
        num_of_inital_training_set = 200

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
            if (i+1)%1000 == 0:
                S_next = self.jacobian_augmentation(f_prime_model, S, O_S)
                S = torch.from_numpy(S_next).to(self.device)

            # Evaluate (DEBUG)
            if (i+1)%100 == 0:
                self.evaluate(f_prime_model, f)

        # Obtain f' & Evaluate f'
        # PENDING
        pass 


