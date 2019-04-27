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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.forward_model = None
        self.test_loader = None
        # can we use cross-entropy?
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
        loss1 = self.loss1(inp1, tar1)

        # replace the weights of self.forward_model (uncompleted)
        forward_model_dict = self.forward_model.state_dict()
        print(forward_model_dict)
        # forward fixed query inputs to the forward model (uncompleted)

        # outputs -> predictions (uncompleted)

        # compare, remember to assign values to predictions 
        predictions = None
        loss2 = self.loss2(predictions, tar2)
        combined_loss = loss1 + loss2
        return combined_loss

    def test(self):
        # predict: expected accurancy around 10% (completed)
        path_string = 'temporary_model.pt'
        torch.save(self.forward_model.state_dict(), path_string)
        model = torch.load(path_string)

        input_size = 784
        for _ in range(1):
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (images, labels) in enumerate(self.test_loader):
                    images = images.reshape(-1, input_size).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.forward_model.forward(images)
                    _, predictions = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
                print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
          
            # reset weights of the forward model 
            self.weight_reset()

        # it looks workable
        # -> load the entire weights, which should be (1, 50890)
        import experiment_interface
        interface = experiment_interface.ExperimentInterface()
        num_of_model_extracted = 1
        weights_dataset = interface.extract_whitebox_model_weights(num_of_model_extracted)
        outputs_dataset = interface.extract_whitebox_model_outputs(num_of_model_extracted)
        predictions_dataset = interface.extract_whitebox_model_predictions(num_of_model_extracted)

        weights, outputs, predictions = weights_dataset[0], outputs_dataset[0], predictions_dataset[0]
        print(weights.shape, outputs.shape, predictions.shape)

        # -> separate weights into 4 parts
        input_size, hidden_size, output_size = 784, 64, 10
        size_W1, size_B1 = input_size * hidden_size, hidden_size
        size_W2, size_B2 = hidden_size * output_size, output_size
        W1_offset = 0
        B1_offset = W1_offset + size_W1
        W2_offset = B1_offset + size_B1
        B2_offset = W2_offset + size_W2
        W1 = weights[W1_offset:W1_offset+size_W1]
        B1 = weights[B1_offset:B1_offset+size_B1]
        W2 = weights[W2_offset:W2_offset+size_W2]
        B2 = weights[B2_offset:B2_offset+size_B2]
        W1 = W1.reshape(-1, input_size)
        W2 = W2.reshape(-1, hidden_size)

        W1 = torch.from_numpy(np.float32(W1)).to(self.device)
        # B1 = torch.from_numpy(np.float32(B1)).to(self.device)
        B1 = torch.from_numpy(np.float32(B1))

        W2 = torch.from_numpy(np.float32(W2)).to(self.device)
        B2 = torch.from_numpy(np.float32(B2)).to(self.device)

        new_weights = []
        new_weights.append(W1)
        new_weights.append(B1)
        new_weights.append(W2)
        new_weights.append(B2)

        print()
        for idx, (name, weights) in enumerate(model.items()):
            if idx == 1:
                print('Original value', weights[0])
                print('New value', new_weights[idx][0])
        print()
        # -> individually load 4 parts
        copy_state_dict = self.forward_model.state_dict()

        for idx, (name, _) in enumerate(model.items()):
            copy_state_dict[name] = new_weights[idx]

        self.forward_model.load_state_dict(copy_state_dict) # great

        # next target: load weights of trained model to achieve test accurancy > 90%

        input_size = 784
        for _ in range(1):
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (images, labels) in enumerate(self.test_loader):
                    images = images.reshape(-1, input_size).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.forward_model.forward(images)
                    _, predictions = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()
                print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
          

    def weight_reset(self):
        self.forward_model.apply(self.reset_func)

    def reset_func(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()



