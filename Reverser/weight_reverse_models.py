import torch
import torch.nn as nn
from torch.nn import functional as F

class FC_WeightModel(nn.Module):
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


class VAE_WeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100000, 800)
        self.fc21 = nn.Linear(800, 200)
        self.fc22 = nn.Linear(800, 200)
        self.fc3 = nn.Linear(200, 800)
        self.fc4 = nn.Linear(800, 50890)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 100000))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class FC_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, inp1, tar1):
        return self.loss(inp1, tar1) 


class VAE_Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, predicted_x, x, mu, logvar):
        # min_r_x, min_x = torch.min(predicted_x), torch.min(x)
        # range_r_x, range_x = torch.max(predicted_x) - min_r_x, torch.max(x) - min_x
        # BCE = F.binary_cross_entropy((predicted_x-min_r_x)/range_r_x, (x-min_x)/range_x, reduction='sum')

        # see Appendix B from VAE paper:
	    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	    # https://arxiv.org/abs/1312.6114
	    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = self.loss(predicted_x, x)
        return loss
