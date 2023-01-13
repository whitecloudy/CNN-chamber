from __future__ import print_function
import ArgsHandler
import torch
import torch.optim as optim
import time
from torch.optim.lr_scheduler import StepLR
from BeamDataset import DatasetHandler, prepare_dataset, calculate_mmse
from Cosine_sim_loss import complex_cosine_sim_loss as cos_loss
from Cosine_sim_loss import make_complex

from ModelHandler import model_selector, Net, Net_transformer_encoder, Net_with1d, Net_withoutLS, Net_withoutRow

import numpy as np
import csv
import copy
import multiprocessing as multi
from CachefileHandler import save_cache, load_cache
from DataExchanger import DataExchanger


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

    model = model_selector(args.model, args.W).to(device)

    testing_model(args, model, device)


if __name__ == '__main__':
    main()
