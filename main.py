from __future__ import print_function
import ArgsHandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from BeamDataset import DatasetHandler
from Cosine_sim_loss import complex_cosine_sim_loss as cos_loss
import numpy as np
import csv

class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model
        model = model % 2
        if model == 0:
            self.conv1 = nn.Conv2d(2, 32, 4, 1) #input is 27 * 8 * 2
            self.conv2 = nn.Conv2d(32, 64, 4, 1) # input is 24 * 5 * 32
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.batch1 = nn.BatchNorm2d(32)
            self.batch2 = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(21 * 2 * 64, 256) # 21 * 2 * 64
            self.fc2 = nn.Linear(256, 12)
        elif model == 1:
            self.conv1 = nn.Conv2d(2, 64, 4, 1) #input is 27 * 8 * 2
            self.conv2 = nn.Conv2d(64, 128, 4, 1) # input is 24 * 5 * 32
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.batch1 = nn.BatchNorm2d(64)
            self.batch2 = nn.BatchNorm2d(128)
            self.fc1 = nn.Linear(21 * 2 * 128, 1024) # 21 * 2 * 64
            self.fc2 = nn.Linear(1024, 12)


    def forward(self, x):
        model = int(self.model/2)

        if model == 0:
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
            x = F.tanh(x)
            #output = F.log_softmax(x, dim=1)
        elif model == 1:
            x = self.conv1(x)
            x = self.batch1(x)
            x = F.leaky_relu(x)
            x = self.conv2(x)
            x = self.batch2(x)
            x = F.leaky_relu(x)
            #x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.leaky_relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.tanh(x)
            #output = F.log_softmax(x, dim=1)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, heur) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.mse_loss(output, target)
        loss = cos_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def get_loss(output, target):
    sum_list = torch.sum(torch.pow(output - target, 2), 1).tolist()
    #target = torch.sum(torch.pow(target, 2), 1)

    #sum_list = torch.div(diff, target).tolist()
    unable_c = 0

    for i, sum_data in enumerate(sum_list):
       while np.isinf(sum_list[i]):
            unable_c += 1
            del sum_list[i]

            if len(sum_list) <= i:
                break

    sum_e = np.sum(sum_list)/(len(output) - unable_c)

    return sum_e, unable_c


def test(model, device, train_loader, test_loader):
    model.eval()
    test_loss = 0
    train_loss = 0
    test_heur_loss = 0
    train_heur_loss = 0
    test_unable_heur = 0
    train_unable_heur = 0

    total_train = 0
    total_test = 0

    with torch.no_grad():
        for data, target, heur in test_loader:
            data, target = data.to(device), target.to(device)
            heur = heur.to(device)

            output = model(data)

            """
            tmp_loss, temp = get_loss(output, target)
            test_loss += tmp_loss
            
            tmp_loss, temp = get_loss(heur, target)
            test_heur_loss += tmp_loss
            test_unable_heur += temp

            total_test += len(data)
            """
            test_loss += cos_loss(output, target)
            test_heur_loss += cos_loss(heur, target)

            # test_loss += F.mse_loss(output, target, reduction='mean').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print(target)
            # print(len(pred))
            # print(target.view_as(pred))
            # correct += pred.eq(target.view_as(pred)).sum().item()

    with torch.no_grad():
        for data, target, heur in train_loader:
            data, target = data.to(device), target.to(device)
            heur = heur.to(device)

            output = model(data)
            train_loss += cos_loss(output, target)
            train_heur_loss += cos_loss(heur, target)

            """

            tmp_loss, temp = get_loss(output, target)
            train_loss += tmp_loss

            tmp_loss, temp = get_loss(heur, target)
            train_heur_loss += tmp_loss
            train_unable_heur += temp

            total_train += len(data)
            """

    test_loss /= len(test_loader)
    train_loss /= len(train_loader)
    test_heur_loss /= len(test_loader)
    train_heur_loss /= len(train_loader)
    #test_unable_heur /= total_test
    #train_unable_heur /= total_train

    print('\nTrain set: Average loss: {:.6f}, Huristic Average Loss: {:.6f}, Unable heur : {:.2f}%'.format(
        train_loss, train_heur_loss, train_unable_heur*100))

    print('\nValidation set: Average loss: {:.6f}, Huristic Average Loss: {:.6f}, Unable heur : {:.2f}%\n'.format(
        test_loss, test_heur_loss, test_unable_heur*100))

    return float(train_loss), float(test_loss), float(train_heur_loss), float(test_heur_loss), train_unable_heur, test_unable_heur



def main():
    ArgsHandler.init_args()
    args = ArgsHandler.args

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

    model = Net(args.model).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.log != None:
        logfile = open(args.log, "w")
        logCSV = csv.writer(logfile)
        logCSV.writerow(["epoch", "train loss", "test loss", "train heuristic loss", "test heuristic loss", "train unable count", "test unable count"])
    else:
        logfile = None
        logCSV = None


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss, test_loss, train_heur_loss, test_heur_loss, train_unable, test_unable = test(model, device, train_loader, test_loader)
        scheduler.step()

        if logCSV is not None:
            logCSV.writerow([epoch, train_loss, test_loss, train_heur_loss, test_heur_loss, train_unable, test_unable])

        if epoch is args.epochs:
            break

        # renew dataset
        dataset_handler.renew_dataset()
        training_dataset = dataset_handler.training_dataset
        test_dataset = dataset_handler.test_dataset
        
        train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    if logfile is not None:
        logfile.close()

    #if args.save_model:
    #    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
