import torch 
import torch.nn as nn
import torchvision
import numpy as np

def separate_predicted_weights(predicted_weights, device):
    predicted_weights = predicted_weights.cpu().detach().numpy().flatten()
    input_size, hidden_size, output_size = 784, 64, 10

    # Determine size of each part
    size_W1, size_B1 = input_size * hidden_size, hidden_size
    size_W2, size_B2 = hidden_size * output_size, output_size
    
    # Determine offset of each part 
    W1_offset = 0
    B1_offset = W1_offset + size_W1
    W2_offset = B1_offset + size_B1
    B2_offset = W2_offset + size_W2

    # Slice each part according to corresponding offset and size 
    W1 = predicted_weights[W1_offset:W1_offset+size_W1]
    B1 = predicted_weights[B1_offset:B1_offset+size_B1]
    W2 = predicted_weights[W2_offset:W2_offset+size_W2]
    B2 = predicted_weights[B2_offset:B2_offset+size_B2]
    
    # Reshape 1-D sliced array if needed
    W1 = W1.reshape(-1, input_size)
    W2 = W2.reshape(-1, hidden_size)
    
    # Transform from numpy to tensor & Move from CPU to GPU
    W1 = torch.from_numpy(np.float32(W1)).to(device)
    B1 = torch.from_numpy(np.float32(B1)).to(device)
    W2 = torch.from_numpy(np.float32(W2)).to(device)
    B2 = torch.from_numpy(np.float32(B2)).to(device)
    
    # Insert each part in a list & Return the list 
    predicted_model_weights = []
    predicted_model_weights.append(W1)
    predicted_model_weights.append(B1)
    predicted_model_weights.append(W2)
    predicted_model_weights.append(B2)
    return predicted_model_weights

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

class PredictionsSimilarityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.forward_model = None
        self.test_loader = None

        # init forward model & test loader 
        self.init_forward_model()
        self.init_test_loader()

    def init_forward_model(self):
        input_size, hidden_size, output_size = 784, 64, 10
        forward_model = WhiteboxNeuralNet(input_size, hidden_size, output_size)   
        forward_model = forward_model.to(self.device) 
        self.forward_model = forward_model

    def init_test_loader(self):
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

    def load_weights_to_forward_model(self, predicted_model_weights):
        copy_state_dict = self.forward_model.state_dict()

        # Itername name of each part in model & Load corresponding predicted model weight to each part 
        for idx, name in enumerate(self.forward_model.state_dict().keys()):
            copy_state_dict[name] = predicted_model_weights[idx]

        self.forward_model.load_state_dict(copy_state_dict) 

    def weight_reset(self):
        self.forward_model.apply(self.reset_func)

    def reset_func(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def compute_single_predicted_predictions_similarity_loss(self, ground_truth_predictions):
        input_size = 784
        with torch.no_grad():
            correct = 0 #DEBUG
            match = 0
            total = 0
            for _, (inputs, labels) in enumerate(self.test_loader):
                inputs = inputs.reshape(-1, input_size).to(self.device)
                ground_truth_predictions = torch.from_numpy(np.int64(ground_truth_predictions)).to(self.device)
                predicted_outputs = self.forward_model.forward(inputs)
                _, predicted_predictions = torch.max(predicted_outputs.data, 1) # this line may be cancelled in the future

                total += ground_truth_predictions.size(0)
                match += (predicted_predictions == ground_truth_predictions).sum().item()

                '''
                DEBUG purpose
                '''
                labels = labels.to(self.device)
                correct += (predicted_predictions == labels).sum().item() 

        similarity = match / total
        predictions_similarity_loss = 1 - similarity

        '''
        DEBUG purpose
        '''
        accurancy = correct / total 
        print('similarity:', round(similarity, 3), 'accurancy:', round(accurancy, 3)) 

        return predictions_similarity_loss 

    def verify_predictions_diff(self, predicted_weights, ground_truth_predictions):
        cumulative_cross_entropy_loss = 0
        cumulative_similarity_loss = 0
        
        batch_size = predicted_weights.shape[0]
        for i in range(batch_size):
            single_predicted_weights, single_ground_truth_predictions = predicted_weights[i], ground_truth_predictions[i]

            # Reshape 1-D weight array into a list
            ''' Reshape 1-D weight array into a list
            A list contains weights of different parts in neural networks. 
            - It follows the order W1, B1, W2, B2, ..., Wi, Bi, ...
            - If layer1 is a FC layer, the shape of W1 would be (hidden_size_1, input_size) and the shape of B1 would be (hidden_size_1, )
            '''
            single_predicted_model_weights = separate_predicted_weights(single_predicted_weights, self.device)

            # Load weights and biasesd in the forward model by predicted model weights 
            self.load_weights_to_forward_model(single_predicted_model_weights)
            
            # Load predicted weights to see the accurancy and loss 
            similarity_loss = self.compute_single_predicted_predictions_similarity_loss(single_ground_truth_predictions)
            cumulative_similarity_loss += similarity_loss

            self.weight_reset()

        average_similarity_loss = cumulative_similarity_loss / batch_size
        print('average_similarity_loss', average_similarity_loss)
     
