import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
        

def VAE_reconstruction_loss(recon_x, x, mu, logvar):

    BCE = reconstruction_function(recon_x, x.view(-1, 784))
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

def generator_loss_function(recon_x, x, mu, logvar, score, label, prediction, is_real, dis_score, alpha, beta):
    import torch # lazy import
    # l1
    BCE = reconstruction_function(recon_x.view(-1, 784), x.view(-1, 784))
    
    # l2
    label_mask = torch.zeros(1, 10) 
    label_mask[0][label.data] = 1
    B, S = score.shape[0], score.shape[1]
    classification_loss = torch.bmm(score.view(B, 1, S), label_mask.view(B, S, 1)).reshape(-1)
    
    if label != prediction:
        alpha = 0

    # l3         
    if is_real:
        discriminator_loss = dis_score
    else:
        discriminator_loss = 1/dis_score
    
    l1 = BCE
    l2 = alpha * classification_loss
    l3 = beta * discriminator_loss
    return (l1+l2+l3), l1, l2, l3