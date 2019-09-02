import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

PATH_SAMPLE_PYTORCH_CNN_MNIST = 'tests/sample_model_PYTORCH_CNN_MNIST.pt'
PATH_SAMPLE_PYTORCH_CNN_CIFAR10 = 'tests/sample_model_PYTORCH_CNN_CIFAR10.pt'

class PytorchCNN_MNIST(nn.Module):
    def __init__(self, num_of_channels, width, height):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class PytorchCNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_target_blackbox_model_MNIST():
    num_of_channels, width, height = 1, 28, 28
    model = PytorchCNN_MNIST(num_of_channels, width, height)
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_CNN_MNIST))
    return model

def load_target_blackbox_model_CIFAR10():
    model = PytorchCNN_CIFAR10()
    model.load_state_dict(torch.load(PATH_SAMPLE_PYTORCH_CNN_CIFAR10))
    return model

def get_experimental_input(num_of_sample, name_of_dataset):
    if name_of_dataset == 'MNIST':
        import torch
        import torchvision
        import torchvision.transforms as transforms
        MNIST_TRANSFORM = transforms.ToTensor()
        dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=MNIST_TRANSFORM, download=True)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=num_of_sample, shuffle=False)
        
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(loader):
                return (inputs.reshape(-1, 1, 28, 28), labels)
    else:
        print('Unsupported dataset:', name_of_dataset)

def display_images(G, inputs, num_of_image_displayed):
    import math
    import matplotlib.pyplot as plt
    for i in range(len(inputs)):
        if i < num_of_image_displayed:
            data = Variable(inputs[i])
            new_data, _, _ = G(data)

            original_img = data.reshape(28, 28)
            img = new_data.detach().numpy().reshape(28, 28)

            plt.subplot(math.ceil(num_of_image_displayed/2), 4, 1+(i*2))
            plt.axis('off')
            plt.imshow(original_img, cmap='gray')

            plt.subplot(math.ceil(num_of_image_displayed/2), 4, 2+(i*2))
            plt.axis('off')
            plt.imshow(img, cmap='gray')
            
    plt.show()

def get_misclassification_ratio(F, G, inputs, labels):
    misclassification_count = 0
    for i in range(len(inputs)):
        data , label = inputs[i], labels[i]
        data = Variable(data)
        new_data, _, _ = G(data)

        new_data = new_data.reshape(-1, 1, 28, 28)
        score = F.forward(new_data)
        prediction = torch.argmax(score)
        if prediction != label:
            misclassification_count += 1
            
    return misclassification_count / len(inputs)