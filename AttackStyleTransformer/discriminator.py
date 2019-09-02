import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 1) 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.relu(self.fc1(x.view(-1, 784)))
        h2 = self.relu(self.fc2(h1))
        return self.sigmoid(self.fc3(h2))

def discriminator_loss_function(is_real, dis_score):
    if is_real:
        return (1/dis_score)
    else: 
        return dis_score 