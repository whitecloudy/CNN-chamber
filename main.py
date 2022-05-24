from __future__ import print_function
import ArgsHandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from BeamDataset import DatasetHandler, prepare_dataset
from Cosine_sim_loss import complex_cosine_sim_loss as cos_loss
from Cosine_sim_loss import make_complex

import numpy as np
import csv
import copy
import multiprocessing as multi
from CachefileHandler import save_cache, load_cache
from DataExchanger import DataExchanger

class Net(nn.Module):
    def __init__(self, row_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1) #input is 9 * 8 * 2
        row_size -= 2
        self.conv2 = nn.Conv2d(64, 64, 3, 1) # input is 7 * 6 * 64
        row_size -= 2
        self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(row_size * 4 * 64 + 256, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.heur_fc1(x1)
        x1 = self.heur_batch(x1)
        x1 = F.leaky_relu(x1)
        x = torch.cat((x, x1), 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x


class Net_with1d(nn.Module):
    def __init__(self, row_size):
        super(Net_with1d, self).__init__()
        #self.conv1 = nn.Conv2d(2, 64, 3, 1) #input is 9 * 8 * 2
        self.first_fc = nn.Linear(16, 6 * 64)
        #row_size -= 2
        self.conv2 = nn.Conv2d(64, 64, 3, 1) # input is 7 * 6 * 64
        row_size -= 2
        self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(row_size * 4 * 64 + 256, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        #torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.first_fc.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = torch.tensor_split(x, 2, dim=1)
        x = torch.cat((x[0], x[1]), dim=3)
        #x = self.conv1(x)
        x = self.first_fc(x)
        x = torch.cat(torch.tensor_split(x, 64, dim=3), dim=1)
        x = self.batch1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.heur_fc1(x1)
        x1 = self.heur_batch(x1)
        x1 = F.leaky_relu(x1)
        x = torch.cat((x, x1), 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x


class Net_withoutRow(nn.Module):
    def __init__(self, row_size):
        super(Net_withoutRow, self).__init__()
        self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = self.heur_fc1(x1)
        x = self.heur_batch(x)
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x

class Net_withoutLS(nn.Module):
    def __init__(self, row_size):
        super(Net_withoutLS, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1) #input is 9 * 8 * 2
        row_size -= 2
        self.conv2 = nn.Conv2d(64, 64, 3, 1) # input is 7 * 6 * 64
        row_size -= 2
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x


def train(args, model, device, train_loader, optimizer, epoch, x_norm, y_norm, do_print=False):
    model.train()
    l = torch.nn.MSELoss(reduction='mean')

    batch_len = int(len(train_loader)/20)

    for batch_idx, (data, heur, target) in enumerate(train_loader):
        data, target, heur = data.to(device), target.to(device), heur.to(device)
        
        data *= x_norm
        target *= y_norm
        heur *= y_norm

        optimizer.zero_grad()

        output = model(data, heur)
        loss = l(output, target)
        #loss = cos_loss(output, target)

        loss.backward()

        #if torch.isnan(loss).any() or torch.isinf(loss).any():
        #    assert False, "Nan is detected"

        optimizer.step()

        if batch_idx % args.log_interval == 0 and do_print:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), batch_len * len(data),
                100. * batch_idx / batch_len, loss.item()))
        if args.dry_run:
            break

        if batch_len <= batch_idx:
            break


def calculate_mmse(data, C_h, C_w):
    data_split = torch.tensor_split(data, (6, 7), dim=3)
    S = data_split[0]
    y = data_split[1]

    S = torch.tensor_split(S, 2, dim=1)
    S_t = torch.tensor(S[0] + S[1]*(1j), dtype=torch.complex128).clone().detach()
    S_t = torch.squeeze(S_t, dim=1)

    y = torch.tensor_split(y, 2, dim=1)
    y_t = torch.tensor(y[0] + y[1]*(1j), dtype=torch.complex128).clone().detach()
    y_t = torch.squeeze(y_t, dim=1)

    SH = torch.conj(torch.transpose(S_t, 2, 1))
    C_h_SH = torch.matmul(C_h, SH)
    S_Ch_SH_minus_Cw_inv = torch.inverse(torch.matmul(torch.matmul(S_t, C_h), SH) + C_w)
    #C_h_SH = SH
    #S_Ch_SH_minus_Cw_inv = torch.inverse(torch.matmul(S, SH))
    
    mmse_result = torch.matmul(C_h_SH, S_Ch_SH_minus_Cw_inv)
    #mmse_result = torch.matmul(S_Ch_SH_minus_Cw_inv, C_h_SH)
    h_hat = torch.squeeze(torch.matmul(mmse_result, y_t))

    return h_hat


def test(model, device, test_loader, x_norm, y_norm, mmse_para, do_print=False):
    model.eval()
    test_loss = torch.tensor(0.0, device=device)
    test_heur_loss = torch.tensor(0.0, device=device)
    test_mmse_loss = torch.tensor(0.0, device=device)

    test_cos_loss = torch.tensor(0.0, device=device)
    test_heur_cos_loss = torch.tensor(0.0, device=device)
    test_mmse_cos_loss = torch.tensor(0.0, device=device)

    test_unable_heur = 0

    batch_len = len(test_loader)/10

    l = torch.nn.MSELoss(reduction='mean')
    
    with torch.no_grad():
        for batch_idx, (data, heur, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            heur = heur.to(device)
            
            data *= x_norm
            heur *= y_norm

            output = model(data, heur)
            output /= y_norm

            test_loss += l(output, target)
            test_cos_loss += cos_loss(output, target)

            data /= x_norm
            heur /= y_norm
            
            #mmse = torch.transpose(torch.mm(mmse_para, torch.transpose(make_complex(heur), 0, 1)), 0, 1)
            #mmse = torch.cat((mmse.real, mmse.imag), 1)
            mmse = calculate_mmse(data, mmse_para[0], mmse_para[1])
            mmse = torch.cat((mmse.real, mmse.imag), dim=1)

            test_heur_loss += l(heur, target)
            test_mmse_loss += l(mmse, target)

            test_heur_cos_loss += cos_loss(heur, target)
            test_mmse_cos_loss += cos_loss(mmse, target)

            if batch_len <= batch_idx:
                break

    test_loss = test_loss.cpu()
    test_heur_loss = test_heur_loss.cpu()
    test_mmse_loss = test_mmse_loss.cpu()

    test_cos_loss = test_cos_loss.cpu()
    test_heur_cos_loss = test_heur_cos_loss.cpu()
    test_mmse_cos_loss = test_mmse_cos_loss.cpu()
    
    test_loss /= batch_len
    test_loss = float(test_loss)
    test_cos_loss /= batch_len
    test_cos_loss = float(test_cos_loss)

    test_heur_loss /= batch_len
    test_mmse_loss /= batch_len

    test_heur_cos_loss /= batch_len
    test_mmse_cos_loss /= batch_len

    if do_print:
        print('\nAverage loss: {:.6f}, Huristic Average Loss: {:.6f}, MMSE Average Loss: {:.6f}, Unable heur : {:.2f}%\n'.format(
            test_loss*1000000, test_heur_loss*1000000, test_mmse_loss*1000000, test_unable_heur*100))

    return test_loss, float(test_heur_loss), float(test_mmse_loss), test_cos_loss, float(test_heur_cos_loss), float(test_mmse_cos_loss), test_unable_heur


def training_model(args, model, device, val_data_num, do_print=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}

    if use_cuda:
        cuda_kwargs = {'num_workers': 8,
                       'pin_memory': True, 
                       'persistent_workers': True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset_handler = DatasetHandler(data_div=args.data_div, val_data_num=val_data_num, row_size=args.W)

    training_dataset = dataset_handler.training_dataset
    if do_print:
        print("Training Dataset : ", len(training_dataset))
    test_dataset = dataset_handler.test_dataset
    if do_print:
        print("Test Dataset : ", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load Normalization
    norm_vector = load_cache(args.log + '.norm')
    if norm_vector is None:
        norm_vector = training_dataset.getNormPara()
        save_cache(norm_vector, args.log + '.norm')

    x_norm_vector = norm_vector[0].to(device)
    y_norm_vector = norm_vector[1].to(device)

    # mmse_para = (C_h, C_w)
    mmse_para = training_dataset.getMMSEpara()
    mmse_para = (mmse_para[0].to(device), mmse_para[1].to(device))

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    logCSV = None
    if args.log is not None:
        logfile = open(args.log+'.csv', "w")

        logCSV = csv.writer(logfile)
        logCSV.writerow(["epoch", "train loss", "test loss", "train ls loss", "test ls loss", "train mmse", "test mmse", "train cos loss", "test cos loss", "train ls cos loss", "test ls cos loss", "train cos mmse", "test cos mmse", "train unable count", "test unable count"])
    else:
        logfile = None
    
    min_loss = float('inf')
    opt_model_para = None

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, x_norm_vector, y_norm_vector, do_print)
        
        if do_print:
            print("<< Test Loader >>")
        test_loss, test_heur_loss, test_mmse, test_cos_loss, test_heur_cos_loss, test_mmse_cos, test_unable = test(model, device, test_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)

        if do_print:
            print("<< Train Loader >>")
        train_loss, train_heur_loss, train_mmse, train_cos_loss, train_heur_cos_loss, train_mmse_cos, train_unable = test(model, device, train_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)

        if logCSV is not None:
            logCSV.writerow([epoch, train_loss, test_loss, train_heur_loss, test_heur_loss, train_mmse, test_mmse, train_cos_loss, test_cos_loss, train_heur_cos_loss, test_heur_cos_loss, train_mmse_cos, test_mmse_cos, train_unable, test_unable])

        scheduler.step()

        if epoch is args.epochs:
            break

        if args.save_model and min_loss > test_cos_loss:
            min_loss = test_cos_loss
            opt_model_para = copy.deepcopy(model.state_dict())

        # renew dataset
        training_dataset = dataset_handler.training_dataset
        test_dataset = dataset_handler.test_dataset
        
        train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        if args.dry_run:
            break

    if logfile is not None:
        logfile.close()

    from pathlib import Path

    if args.save_model:
        torch.save(opt_model_para, str(Path.home())+"/data/cache/"+args.log+'.pt')


def inference(model, device, x_data, heur_data, x_norm, y_norm):
    x_data = x_data.to(device)
    heur_data = heur_data.to(device)

    x_data *= x_norm
    heur_data *= y_norm

    output = model(x_data[None, ...], heur_data[None, ...])

    output /= y_norm
    heur_data /= y_norm

    output = output.to('cpu').detach().numpy().reshape((12))
    output = output[0:6] + output[6:12]*1j

    heur_data = heur_data.to('cpu').detach().numpy()
    heur_data = heur_data[0:6] + heur_data[6:12]*1j

    return output, heur_data


def testing_model(args, model, device):
    # Loading model parameter
    with open(args.test+'.pt','rb') as pt_file:
        model.load_state_dict(torch.load(pt_file, map_location=device))

    model.eval()
    x_norm_vector, y_norm_vector = load_cache(args.test+'.norm')
    x_norm_vector = x_norm_vector.to(device)
    y_norm_vector = y_norm_vector.to(device)

    data_exchanger = DataExchanger()

    row_size = args.W

    print("Ready")

    while True:
        x_data, heur_data = data_exchanger.recv_data(row_size)
        if x_data is None:
            break
        
        result, heur_data = inference(model, device, x_data, heur_data, x_norm_vector, y_norm_vector)

        data_exchanger.send_channel(result)
        data_exchanger.send_channel(heur_data)


def main():
    ArgsHandler.init_args()
    args = ArgsHandler.args

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:"+str(args.gpunum) if use_cuda else "cpu")

    """
    if use_cuda:
        if torch.cuda.device_count() >= (args.gpunum-1):
            torch.cuda.set_device(args.gpunum)
        else:
            print("No gpu number")
            exit(1)
    """

    #model_withoutRow = Net_withoutRow(args.W).to(device)
    #model_withoutLS = Net_withoutLS(args.W).to(device)

    if args.test is None:
        prepare_dataset(args.W, 1)
        """
        args_list = []
        for i in range(args.data_div):
            args_list.append((args, device, i))

        with multi.Pool(args.data_div) as p:
            p.map(training_worker, args_list)
        """
        #model1 = Net(args.W).to(device)
        #model2 = Net_withoutLS(args.W).to(device)
        #model3 = Net_withoutRow(args.W).to(device)
        model4 = Net_with1d(args.W).to(device)

        training_model(args, model4, device, args.val_data_num, True)
    else:
        model = Net(args.W).to(device)
        testing_model(args, model, device)


if __name__ == '__main__':
    main()
