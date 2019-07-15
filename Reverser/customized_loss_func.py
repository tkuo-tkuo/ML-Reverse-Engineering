import torch 
import torch.nn as nn
import torchvision
import numpy as np


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

class CustomerizedLoss(nn.Module):
    def __init__(self):
        super(CustomerizedLoss, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"        
        
        self.forward_model = None
        self.test_loader = None

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.CrossEntropyLoss()

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

    def forward(self, inp1, tar1, inp2, tar2):
        loss1 = self.loss1(inp1, tar1) * 20
        loss2 = self.predictions_similarity_loss(inp2, tar2)
        # print('loss 1', loss1, 'loss 2', loss2)
        # print()
        combined_loss = loss1 + loss2
        return combined_loss, loss1, loss2

    def predictions_similarity_loss(self, predicted_weights, ground_truth_predictions):
        cumulative_cross_entropy_loss = 0
        cumulative_similarity_loss = 0
        
        batch_size = predicted_weights.shape[0]
        for i in range(batch_size):
            single_predicted_weights, single_ground_truth_predictions = predicted_weights[i], ground_truth_predictions[i]

            # Reshape 1-D weight array into a list
            '''
            A list contains weights of different parts in neural networks. 
            - It follows the order W1, B1, W2, B2, ..., Wi, Bi, ...
            - If layer1 is a FC layer, the shape of W1 would be (hidden_size_1, input_size) and the shape of B1 would be (hidden_size_1, )
            '''
            single_predicted_model_weights = self.separate_predicted_weights(single_predicted_weights)

            # Load weights and biasesd in the forward model by predicted model weights 
            self.load_weights_to_forward_model(single_predicted_model_weights)
            
            # Load predicted weights to see the accurancy and loss 
            similarity_loss, cross_entropy_loss = self.compute_single_predicted_predictions_similarity_loss(single_ground_truth_predictions)
            cumulative_cross_entropy_loss += cross_entropy_loss
            cumulative_similarity_loss += similarity_loss

            self.weight_reset()

        average_cross_entropy_loss = cumulative_cross_entropy_loss / batch_size
        average_similarity_loss = cumulative_similarity_loss / batch_size
        # print('average_similarity_loss', average_similarity_loss)
        return average_cross_entropy_loss

    def separate_predicted_weights(self, predicted_weights):
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
        W1 = torch.from_numpy(np.float32(W1)).to(self.device)
        B1 = torch.from_numpy(np.float32(B1)).to(self.device)
        W2 = torch.from_numpy(np.float32(W2)).to(self.device)
        B2 = torch.from_numpy(np.float32(B2)).to(self.device)
        
        # Insert each part in a list & Return the list 
        predicted_model_weights = []
        predicted_model_weights.append(W1)
        predicted_model_weights.append(B1)
        predicted_model_weights.append(W2)
        predicted_model_weights.append(B2)
        return predicted_model_weights

    def load_weights_to_forward_model(self, predicted_model_weights):
        copy_state_dict = self.forward_model.state_dict()

        # Itername name of each part in model & Load corresponding predicted model weight to each part 
        for idx, name in enumerate(self.forward_model.state_dict().keys()):
            copy_state_dict[name] = predicted_model_weights[idx]

        self.forward_model.load_state_dict(copy_state_dict) 

    def predicted_predictions_loss(self):
        input_size = 784
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (images, labels) in enumerate(self.test_loader):
                images = images.reshape(-1, input_size).to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward_model.forward(images)
                _, predictions = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accurancy = correct / total 
        mis_classified_predictions_ratio = 1 - accurancy
        return accurancy, mis_classified_predictions_ratio  

    def compute_single_predicted_predictions_similarity_loss(self, ground_truth_predictions):
        input_size = 784
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (images, _) in enumerate(self.test_loader):
                images = images.reshape(-1, input_size).to(self.device)
                ground_truth_predictions = torch.from_numpy(np.int64(ground_truth_predictions)).to(self.device)
                predicted_outputs = self.forward_model.forward(images)
                _, predicted_predictions = torch.max(predicted_outputs.data, 1) # this line may be cancelled in the future

                cross_entropy_loss = self.loss2(predicted_outputs, ground_truth_predictions)
                # print(simi_ls)

                total += ground_truth_predictions.size(0)
                correct += (predicted_predictions == ground_truth_predictions).sum().item()

        similarity = correct / total 
        predictions_similarity_loss = 1 - similarity
        return predictions_similarity_loss, cross_entropy_loss        

    def weight_reset(self):
        self.forward_model.apply(self.reset_func)

    def reset_func(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    # there's a problem existing in the predicted_predictions_similarity function

    # to verify predicted_predictions_similarity correctness
    # -> overfit this functions with two identical models to achieve 100% similarity 

    # ultimate goal
    # 1. efficient train l2 loss -> is it apply cross_entropy? 
    # -> is l2 loss actually be trained? -> refer to the white-box model training process!
    # 2. overfitting in the training process
    # 3. verify in the test process


    # Is it acutally be trained during the training process?
    # -> print it console it see whether l2 loss can be trained 
    # Ans: I think that l2 loss is not trained at all -> apply a gradient descent method for l2 loss (see how white-box neural networks do this, crossentropy loss?)
    # -> visualize the training process 
