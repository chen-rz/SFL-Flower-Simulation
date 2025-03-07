import math
import torch
from constants import LEARNING_RATE, MOMENTUM


def train(net, trainloader, epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = 0.0
    mean_square_batch_loss = 0.0
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            train_loss += loss.item()
            mean_square_batch_loss += loss.item() ** 2
            loss.backward()
            optimizer.step()
    
    train_loss = train_loss / epochs / len(trainloader.dataset)
    mean_square_batch_loss = math.sqrt(mean_square_batch_loss / epochs / len(trainloader.dataset))

    return train_loss, mean_square_batch_loss


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (torch.Tensor(predicted == labels)).sum().item()
    loss = loss / len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
