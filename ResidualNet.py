from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn.functional as F

batch_size = 64
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=False,
                               transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=False,
                              transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, bias=False)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, bias=False)
        self.pooling = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.linear = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.pooling(self.conv1(x)))
        x = self.rblock1(x)
        x = F.relu(self.pooling(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    runnning_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        if(batch_idx % 300 == 299):
            print('[%d %5d] loss: %.3f' % (epoch+1, batch_idx+1, runnning_loss/300))
            runnning_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()