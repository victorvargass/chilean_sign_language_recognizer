import torch
from torch.nn import functional as F

"""
Red tipo lenet para partir
"""
class LeNet5(torch.nn.Module):
    
    def __init__(self, n_output=21):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.adapool  = torch.nn.AdaptiveMaxPool2d((5, 5))
        self.fc1 = torch.nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=n_output)
        self.activation = torch.nn.ReLU()
    
    def forward(self, x):
        z = self.activation(self.conv1(x))
        z = F.max_pool2d(z, 2)
        z = self.activation(self.conv2(z))
        #z = F.max_pool2d(z, 2)
        z = self.adapool(z)
        z = z.view(z.size(0), -1)
        z = self.activation(self.fc1(z))
        z = self.activation(self.fc2(z))
        y = self.fc3(z)
        return y
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

class AlexNet(torch.nn.Module):

    def __init__(self, n_output=21):
        super(AlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, n_output),
        )

    def forward(self, x):
        #print(x.shape) torch.Size([1, 3, 224, 224])
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))