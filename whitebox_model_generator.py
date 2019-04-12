import torch 
import torch.nn as nn
import torchvision

'''
functions are requried:
model_extraction
output_extraction

weights comparision
outputs comparision
'''

class WhiteboxModelGenerator():
    
    def __init__(self):
        # If GPU resource is avaiable, use GPU. Otherwise, use CPU. 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.loss_func = None
        self.optimizer = None

        # Hyper parameters
        self.num_of_epochs = 1
        self.number_of_prinint_interval = 100
        self.input_size = 784

    def generate(self, number_of_white_boxes_generated):
        for generation_idx in range(number_of_white_boxes_generated):
            self.train(generation_idx)
            self.test(generation_idx)
        
            self.weight_reset()

    def weight_reset(self):
        self.model.apply(self.reset_func)

    def reset_func(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def set_generation_hyperparameters(self, num_of_epochs=1, number_of_prinint_interval=100, input_size=784):
        self.num_of_epochs = num_of_epochs
        self.number_of_prinint_interval = number_of_prinint_interval
        self.input_size = input_size

    def set_dataset_loader(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def set_model(self, model):
        model = model.to(self.device)
        self.model = model      

    def set_loss_func(self, loss_func):
        self.loss_func = loss_func

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, generation_index):
        total_step = len(self.train_loader)
        for epoch in range(self.num_of_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Move tensors to the configured device
                images = images.reshape(-1, self.input_size).to(self.device)
                labels = labels.to(self.device)

                # Forwarding 
                outputs = self.model.forward(images)
                loss = self.loss_func(outputs, labels)

                # Optimization (back-propogation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1) % self.number_of_prinint_interval == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_of_epochs, i+1, total_step, loss.item()))

        self.save_model('./whitebox_database/model'+str(generation_index+1)+'.pt')

    def test(self, generation_index):
        with torch.no_grad():
            correct = 0
            total = 0
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
        

    