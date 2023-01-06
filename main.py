from __future__ import print_function
import ArgsHandler
import torch
import torch.optim as optim
import time
from torch.optim.lr_scheduler import StepLR
from BeamDataset import DatasetHandler, prepare_dataset
from Cosine_sim_loss import complex_cosine_sim_loss as cos_loss
from Cosine_sim_loss import make_complex

from ModelHandler import model_selector, Net, Net_transformer_encoder, Net_with1d, Net_withoutLS, Net_withoutRow

import numpy as np
import csv
import copy
import multiprocessing as multi
from CachefileHandler import save_cache, load_cache
from DataExchanger import DataExchanger



def train(args, model, device, train_loader, optimizer, epoch, x_norm, y_norm, do_print=False):
    model.train()
    l = torch.nn.MSELoss(reduction='mean')

    batch_len = int(len(train_loader)/20)
    batch_multiply_count = 0

    for batch_idx, (data, heur, target) in enumerate(train_loader):
        data, target, heur = data.to(device), target.to(device), heur.to(device)

        if batch_multiply_count == 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_multiply_count = args.batch_multiplier
        
        data *= x_norm
        target *= y_norm
        heur *= y_norm

        # data_split = torch.split(data, args.batch_size, dim=0)
        # target_split = torch.split(target, args.batch_size, dim=0)
        # heur_split = torch.split(heur, args.batch_size, dim=0)
        output = model(data, heur)
        loss = l(output, target) / args.batch_multiplier
        #loss = cos_loss(output, target)

        # optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        batch_multiply_count -= 1

        # for i in range(len(data)):
        #     output = model(data_split[i], heur_split[i])
        #     loss = l(output, target_split[i])
        #     #loss = cos_loss(output, target)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     #if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     #    assert False, "Nan is detected"

        if batch_idx % args.log_interval == 0 and do_print:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), batch_len * len(data),
                100. * batch_idx / batch_len, loss.item()))
        if args.dry_run:
            break

        if batch_len <= batch_idx:
            break


def calculate_mmse(data, C_h, C_w):
    data_dim = data.dim()-1
    data_split = torch.tensor_split(data, (6, 7), dim=(data_dim))
    S = data_split[0]
    y = data_split[1]

    S = torch.tensor_split(S, 2, dim=(data_dim-2))
    S_t = ((S[0] + S[1]*1j).clone().detach()).type(torch.complex128)
    S_t = torch.squeeze(S_t, dim=(data_dim-2))

    y = torch.tensor_split(y, 2, dim=(data_dim-2))
    y_t = ((y[0] + y[1]*1j).clone().detach()).type(torch.complex128)
    y_t = torch.squeeze(y_t, dim=(data_dim-2))

    SH = torch.conj(torch.transpose(S_t, data_dim-2, data_dim-1))
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


def training_model(args, model, device, val_data_num, do_print=False, early_stopping_patience=3):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # train_kwargs = {'batch_size': args.batch_size*args.load_minibatch_multiplier, 'shuffle': True}
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}

    if use_cuda:
        cuda_kwargs = {'num_workers': 8,
                       'pin_memory': True, 
                       'persistent_workers': True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    aug_para = (args.aug1, args.aug2)
    dataset_handler = DatasetHandler(data_div=args.data_div, val_data_num=val_data_num, row_size=args.W, aug_para=aug_para)

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
        test_loss, test_heur_loss, test_mmse, test_cos_loss, test_heur_cos_loss, test_mmse_cos, test_unable = test(model, device, valid_test_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)

        scheduler.step()
        with torch.cuda.device('cuda:'+str(args.gpunum)):
            torch.cuda.empty_cache()

        if do_print:
            print("<< Train Loader >>")
        train_loss, train_heur_loss, train_mmse, train_cos_loss, train_heur_cos_loss, train_mmse_cos, train_unable = test(model, device, train_test_loader, x_norm_vector, y_norm_vector, mmse_para, do_print)
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
        logfile.close()

    from pathlib import Path

    if args.save_model:
        torch.save(opt_model_para, str(Path.home())+"/data/cache/"+args.log+'.pt')


def inference(model, device, x_data, heur_data, x_norm, y_norm, C_h, C_w):
    x_data = x_data.to(device)
    heur_data = heur_data.to(device)

    x_data *= x_norm
    heur_data *= y_norm

    output = model(x_data[None, ...], heur_data[None, ...])

    x_data /= x_norm
    output /= y_norm
    heur_data /= y_norm

    mmse = calculate_mmse(x_data, C_h, C_w).to('cpu').detach().numpy()

    output = output.to('cpu').detach().numpy().reshape((12))
    output = output[0:6] + output[6:12]*1j

    heur_data = heur_data.to('cpu').detach().numpy()
    heur_data = heur_data[0:6] + heur_data[6:12]*1j

    return output, heur_data, mmse


def testing_model(args, model, device):
    # Loading model parameter
    with open(args.test+'.pt','rb') as pt_file:
        model.load_state_dict(torch.load(pt_file, map_location=device))

    model.eval()
    print(args.test)
    x_norm_vector, y_norm_vector = load_cache(args.test+'.norm', testing=True)
    x_norm_vector = x_norm_vector.to(device)
    y_norm_vector = y_norm_vector.to(device)

    # mmse_para = (C_h, C_w)
    C_h, C_w = load_cache(args.test + '.mmse', testing=True)
    C_h = C_h.to(device)
    C_w = C_w.to(device)

    row_size = args.W

    data_exchanger = DataExchanger(port=(11039+row_size))

    print("Ready")

    while True:
        x_data, heur_data, select_data = data_exchanger.recv_data(row_size)
        if x_data is None:
            break
        
        result, heur_data, mmse = inference(model, device, x_data, heur_data, x_norm_vector, y_norm_vector, C_h, C_w)

        select_data = select_data.to('cpu').detach().numpy()
        select_data = select_data[0:6] + select_data[6:12]*1j

        data_exchanger.send_channel(result)
        data_exchanger.send_channel(heur_data)
        data_exchanger.send_channel(mmse)
        data_exchanger.send_channel(select_data)

        rt_val = data_exchanger.wait_reset()
        if rt_val == -1:
            break


def main():
    ArgsHandler.init_args()
    args = ArgsHandler.args

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    print("Connect GPU : ", args.gpunum)

    device = torch.device("cuda:"+str(args.gpunum) if use_cuda else "cpu")

    """
    if use_cuda:
        if torch.cuda.device_count() >= (args.gpunum-1):
            torch.cuda.set_device(args.gpunum)
        else:
            print("No gpu number")
            exit(1)
    """


    if args.test is None:
        prepare_dataset(args.W, 1, args.dry_run)
        """
        args_list = []
        for i in range(args.data_div):
            args_list.append((args, device, i))

        with multi.Pool(args.data_div) as p:
            p.map(training_worker, args_list)
        """

        print(args.model)
        model = model_selector(args.model, args.W).to(device)

        training_model(args, model, device, args.val_data_num, True, early_stopping_patience=args.patience)
    else:
        model = model_selector(args.model, args.W).to(device)

        testing_model(args, model, device)


if __name__ == '__main__':
    main()
