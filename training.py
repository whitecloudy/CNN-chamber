from __future__ import print_function
import ArgsHandler
import torch
import torch.optim as optim
import time
from torch.optim.lr_scheduler import StepLR
from BeamDataset import DatasetHandler, prepare_dataset, calculate_mmse
from Cosine_sim_loss import complex_cosine_sim_loss as cos_loss
from Cosine_sim_loss import make_complex

from ModelHandler import model_selector

import csv
import copy
import numpy as np
from CachefileHandler import save_cache, load_cache


def complexTensor2FloatTensor(data):
    return 


def train(args, model, device, train_loader, optimizer, epoch, x_norm, y_norm, do_print=False):
    model.train()
    l = torch.nn.MSELoss(reduction='mean')

    # batch_len = int(len(train_loader)/20)
    batch_len = int(len(train_loader))

    batch_multiply_count = args.batch_multiplier
    optimizer.zero_grad()

    for batch_idx, (data, heur, target) in enumerate(train_loader):
        data, target, heur = data.to(device), target.to(device), heur.to(device)

        if batch_multiply_count == 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_multiply_count = args.batch_multiplier
        
        data *= x_norm
        target *= y_norm
        heur *= y_norm

        output = model(data, heur)
        loss = l(output, target) / args.batch_multiplier
        #loss = cos_loss(output, target)

        loss.backward()
        batch_multiply_count -= 1

        if batch_idx % args.log_interval == 0 and do_print:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), batch_len * len(data),
                100. * batch_idx / batch_len, loss.item()))
        if args.dry_run:
            break

        if batch_len < batch_idx:
            break
    
    optimizer.step()



def validation(model, device, test_loader, x_norm, y_norm, mmse_para, do_print=False):
    model.eval()
    test_loss = torch.tensor(0.0, device=device)
    test_heur_loss = torch.tensor(0.0, device=device)
    test_mmse_loss = torch.tensor(0.0, device=device)

    test_cos_loss = torch.tensor(0.0, device=device)
    test_heur_cos_loss = torch.tensor(0.0, device=device)
    test_mmse_cos_loss = torch.tensor(0.0, device=device)

    test_unable_heur = 0

    batch_len = len(test_loader)

    l = torch.nn.MSELoss(reduction='mean')
    
    with torch.no_grad():
        for batch_idx, (data, heur, target) in enumerate(test_loader):
            data, target, heur = data.to(device), target.to(device), heur.to(device)
            
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


def training_model(args, model, device, val_data_num, do_print=False, early_stopping_patience=3):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}

    if use_cuda:
        cuda_kwargs = {'num_workers': 16,
                       'pin_memory': True, 
                       'persistent_workers': True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset_handler = DatasetHandler(data_div=args.data_div, val_data_num=val_data_num, row_size=args.W, aug_ratio=args.aug_ratio)

    training_dataset = dataset_handler.training_dataset
    if do_print:
        print("Training Dataset : ", len(training_dataset))
    training_test_dataset = dataset_handler.training_test_dataset
    if do_print:
        print("training Test Dataset : ", len(training_test_dataset))
    test_dataset = dataset_handler.test_dataset
    if do_print:
        print("Test Dataset : ", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
    train_test_loader = torch.utils.data.DataLoader(training_test_dataset, **test_kwargs)
    valid_test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load Normalization
    norm_vector = load_cache(args.log + '.norm')
    if norm_vector is None:
        norm_vector = training_dataset.getNormPara()
        save_cache(norm_vector, args.log + '.norm')

    x_norm_vector = norm_vector[0].to(device)
    y_norm_vector = norm_vector[1].to(device)

    # mmse_para = (C_h, C_w)
    mmse_para = load_cache(args.log + '.mmse')
    if mmse_para is None:
        mmse_para = training_dataset.getMMSEpara()
        save_cache(mmse_para, args.log + '.mmse')
    
    mmse_para = (mmse_para[0].to(device), mmse_para[1].to(device))

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    logCSV = None
    if args.log is not None:
        logfile = open("result/" + args.log+'.csv', "w")

        logCSV = csv.writer(logfile)
        logCSV.writerow(["epoch", "train loss", "test loss", "train ls loss", "test ls loss", "train mmse", "test mmse", "train cos loss", "test cos loss", "train ls cos loss", "test ls cos loss", "train cos mmse", "test cos mmse", "train unable count", "test unable count"])
    else:
        logfile = None
    
    min_cos_loss = float('inf')
    min_loss = float('inf')
    early_stopping_ctr = early_stopping_patience
    opt_model_para = None

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch, x_norm_vector, y_norm_vector, do_print)
        end_time = time.time()

        consumed_time = end_time - start_time

        if do_print:
            print("Training Consumed time: ", consumed_time)

        start_time = time.time()
        if do_print:
            print("<< Test Loader >>")
        test_loss, test_heur_loss, test_mmse, test_cos_loss, test_heur_cos_loss, test_mmse_cos, test_unable = validation(model, device, valid_test_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)

        scheduler.step()
        with torch.cuda.device('cuda:'+str(args.gpunum)):
            torch.cuda.empty_cache()

        if do_print:
            print("<< Train Loader >>")
        train_loss, train_heur_loss, train_mmse, train_cos_loss, train_heur_cos_loss, train_mmse_cos, train_unable = validation(model, device, train_test_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)
        end_time = time.time()

        consumed_time = end_time - start_time

        if do_print:
            print("Validation Consumed time: ", consumed_time)

        if logCSV is not None:
            logCSV.writerow([epoch, train_loss, test_loss, train_heur_loss, test_heur_loss, train_mmse, test_mmse, train_cos_loss, test_cos_loss, train_heur_cos_loss, test_heur_cos_loss, train_mmse_cos, test_mmse_cos, train_unable, test_unable])


        if epoch is args.epochs:
            break

        if args.save_model and min_cos_loss > test_cos_loss:
            min_cos_loss = test_cos_loss
            opt_model_para = copy.deepcopy(model.state_dict())
        
        if min_loss > test_loss:
            min_loss = test_loss
            early_stopping_ctr = early_stopping_patience
        else:
            early_stopping_ctr -= 1
            if early_stopping_ctr <= 0:
                print("Early stopping Triggered")
                break

        # renew dataset
        # training_dataset = dataset_handler.training_dataset
        # test_dataset = dataset_handler.test_dataset
        # training_test_dataset = dataset_handler.training_test_dataset
        
        # train_loader = torch.utils.data.DataLoader(training_dataset, **train_kwargs)
        # train_test_loader = torch.utils.data.DataLoader(training_test_dataset, **test_kwargs)
        # valid_test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        if args.dry_run:
            break

    if logfile is not None:
        logfile.writerow("FIN")
        logfile.close()

    from pathlib import Path

    if args.save_model:
        torch.save(opt_model_para, "cache/"+args.log+'.pt')


def main():
    ArgsHandler.init_args()
    args = ArgsHandler.args

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print("Connect GPU : ", args.gpunum)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed+256)
    
    if use_cuda:
        if torch.cuda.device_count() >= (args.gpunum-1):
            torch.cuda.set_device(args.gpunum)
        else:
            print("No gpu number")
            exit(1)

    prepare_dataset(args.W, 1, args.dry_run)

    device = torch.device("cuda:"+str(args.gpunum) if use_cuda else "cpu")

    print(args.model)
    model = model_selector(args.model, args.W).to(device)

    training_model(args, model, device, args.val_data_num, True, early_stopping_patience=args.patience)


if __name__ == '__main__':
    main()
