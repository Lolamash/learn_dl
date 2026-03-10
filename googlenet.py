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

class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, 1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, 1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, 5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, 1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, 3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, 3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)#按channel维度进行拼接

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, bias=False)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5, bias=False)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.pooling = torch.nn.MaxPool2d(2)
        self.linear = torch.nn.Linear(1408, 10)

    def forward(self, x):
        x = F.relu(self.pooling(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.pooling(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        #print(x.shape())
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