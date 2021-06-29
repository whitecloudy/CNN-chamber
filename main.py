from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from BeamDataset import DatasetHandler
import numpy as np
import csv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 4, 1) #input is 27 * 8 * 2
        self.conv2 = nn.Conv2d(32, 64, 4, 1) # input is 24 * 5 * 32
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(21 * 2 * 64, 256) # 21 * 2 * 64
        self.fc2 = nn.Linear(256, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def get_loss(output, target):
    """
    sum_e = 0.0

    for i, output_data in enumerate(output):
        target_data = target[i]

        output_c = []
        target_c = []

        for c in range(0, len(output_data), 2):
            target_complex = complex(target_data[c], target_data[c+1])
            output_complex = complex(output_data[c], output_data[c+1])

            output_c.append(output_complex)
            target_c.append(target_complex)

        output_data = np.array(output_c)
        target_data = np.array(target_c)

        sum_e += (abs(np.dot(output_data, target_data)))/(np.linalg.norm(output_data) * np.linalg.norm(target_data))
    """
    diff = output - target

    sum_e = 0.0

    for i, diff_data in enumerate(diff):
        target_data = target[i]
        output_data = output[i]

        # TODO : This should be removed
        # TODO : loss calculation must be improved
        # print(output_data[0], ", ", target_data[0])
        # print(diff_data)
        # print((diff_data ** 2).sum())
        # print((output_data ** 2).sum())
        # print((target_data ** 2).sum())
        # print()

        sum_e += (diff_data ** 2).sum()/(target_data ** 2).sum()

    sum_e /= len(output)
        
    return sum_e


def test(model, device, train_loader, test_loader):
    model.eval()
    test_loss = 0
    train_loss = 0
    correct = 0
    t = None
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += get_loss(output, target)
            # test_loss += F.mse_loss(output, target, reduction='mean').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print(target)
            # print(len(pred))
            # print(target.view_as(pred))
            # correct += pred.eq(target.view_as(pred)).sum().item()

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += get_loss(output, target)
    
    test_loss /= len(test_loader)
    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print('\nValidation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return float(train_loss), float(test_loss)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gpunum', type=int, default=0,
                        help='number of gpu use')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--log', type=str, default=None,
                        help='If log file name given, we write Logs')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'pin_memory': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    if use_cuda:
        if torch.cuda.device_count() >= (args.gpunum-1):
            torch.cuda.set_device(args.gpunum)
        else:
            print("No gpu number")
            exit(1)


        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    """
    #dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                      transform=transform)
    #dataset2 = datasets.MNIST('../data', train=False,
    #                   transform=transform)
    dataset_handler = DatasetHandler()

    training_dataset = dataset_handler.training_dataset
    print("Training Dataset : ", len(training_dataset))
    test_dataset = dataset_handler.test_dataset
    print("Test Dataset : ", len(test_dataset))
    
    train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.log != None:
        logfile = open(args.log, "w")
        logCSV = csv.writer(logfile)
        logCSV.writerow(["epoch", "train loss", "test loss"])
    else:
        logfile = None
        logCSV = None


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss, test_loss = test(model, device, train_loader, test_loader)
        scheduler.step()

        if logCSV is not None:
            logCSV.writerow([epoch, train_loss, test_loss])

        if epoch is args.epochs:
            break

        # renew dataset
        dataset_handler.renew_dataset()
        training_dataset = dataset_handler.training_dataset
        test_dataset = dataset_handler.test_dataset
        
        train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    logfile.close()

    #if args.save_model:
    #    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
