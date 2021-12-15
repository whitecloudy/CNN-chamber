import random
import time
import torch
import numpy as np
import ArgsHandler
import Proc
from CachefileHandler import load_cache, save_cache, make_cache_hashname

from torch.utils.data import Dataset
from DataProcessor import DataProcessor, global_key_list, global_data_handler

class BeamDataset(Dataset):
    def __init__(self, multiply, num_list, data_size=6, normalize=None, MMSE_para=None):
        self.multiply = multiply
        self.data_size = data_size

        cache_filename_list = [(str(data_size)+"_"+str(multiply)+"_"+str(i)+'_20211213.bin', ) for i in num_list]
        self.hashname = make_cache_hashname(cache_filename_list)
        self.data_list = []
        self.idx_list = []

        for idx, filename in enumerate(cache_filename_list):
            loaded_data = load_cache(*filename)

            # data will be loaded in (x, h, y)
            self.data_list.append(loaded_data)
            self.idx_list += [(idx, j) for j in range(len(loaded_data))]
            print(len(self.idx_list))

        self.MMSE_para = MMSE_para
        self.normalize = normalize
        self.renew_data()

    def renew_data(self):
        random.shuffle(self.idx_list)

    def calculate_normalize(self):
        x_mean = 0
        y_mean = 0
        h_mean = 0
        for data in self.data_list:
            for x_mat, h, y in data:
                for x in x_mat:
                    x_mean += abs(np.array(x[6:8]))
                y_mean += abs(np.array(y).reshape((1, 6)))
                h_mean += abs(np.array(h).reshape((1, 6)))

        x_mean /= (len(self.idx_list) * self.data_size)
        x_mean = np.append([1. for i in range(6)], x_mean)
        y_mean /= len(self.idx_list)
        h_mean /= len(self.idx_list)

        return (1/x_mean, 1/h_mean, 1/y_mean)
    
    def calculate_MMSE_parameter(self):
        H_Y_pair = []
        Y_avg_sum = 0
        H_avg_sum = 0

        for data in self.data_list:
            for x_mat, h, y in data:
                Y = np.array(h).reshape((1, 6))
                H = np.array(y).reshape((1, 6))

                H_Y_pair.append((H, Y))

                Y_avg_sum += Y
                H_avg_sum += H

        N = len(self.idx_list)

        mu_y = Y_avg_sum / N
        mu_h = H_avg_sum / N
        r_hy = 0
        r_yy = 0

        for h, y in H_Y_pair:
            h_hat = h - mu_h
            y_hat = y - mu_y
            r_hy += (np.matrix(h_hat).T * np.conj(np.matrix(y_hat)))
            r_yy += (np.matrix(y_hat).T * np.conj(np.matrix(y_hat)))

        r_hy /= len(H_Y_pair)
        r_yy /= len(H_Y_pair)

        r_yy_inv = r_yy.getI()
        mmse = r_hy * r_yy_inv

        print(mmse.shape)

        return mmse


    def getNormPara(self):
        if self.normalize is None:
            cache_name = self.hashname + '.norm'
            self.normalize = load_cache(cache_name)
            if self.normalize is None:
                x, h, y = self.calculate_normalize()
                self.normalize = (torch.FloatTensor(x), torch.FloatTensor([y, y]).reshape(12,))
                save_cache(self.normalize, cache_name)

        return self.normalize

    def getMMSEpara(self):
        if self.MMSE_para is None:
            cache_name = self.hashname + '.mmse'
            self.MMSE_para = load_cache(cache_name)
            if self.MMSE_para is None:
                mmse = self.calculate_MMSE_parameter()
                self.MMSE_para = torch.tensor(mmse, dtype=torch.complex64)
                save_cache(self.MMSE_para, cache_name)

        return self.MMSE_para

    def __len__(self):
        return int(len(self.idx_list)/self.multiply/4)

    def __getitem__(self, idx):
        i, j = self.idx_list[idx]
        x, h, y = self.data_list[i][j]

        x = torch.FloatTensor([x.real, x.imag])
        y = torch.FloatTensor([y.real, y.imag]).reshape(12,)
        h = torch.FloatTensor([h.real, h.imag]).reshape(12,)

        return x, h, y


class DatasetHandler:
    def __init__(self,  multiply=1, data_div=5, val_data_num=1, row_size=6):
        self.multiply = multiply
        self.data_div = data_div
        self.val_data_num = val_data_num
        self.row_size = row_size

        self.training_dataset = None
        self.test_dataset = None
        
        self.prepare_dataset()

    def prepare_dataset(self):
        data_div = self.data_div
        val_data_num = self.val_data_num

        nums_for_training = []
        nums_for_validation = []

        for i in range(data_div):
            step_num_list = list(range(i * int(10/data_div), (i+1) * int(10/data_div)))

            if i == val_data_num:
                nums_for_validation += step_num_list
            else:
                nums_for_training += step_num_list
        
        self.training_dataset = BeamDataset(self.multiply, nums_for_training, self.row_size)#, self.normalize)
        self.normalize = self.training_dataset.getNormPara()
        self.test_dataset = BeamDataset(self.multiply, nums_for_validation, self.row_size, self.normalize)

    def renew_dataset(self):
        self.training_dataset.renew_data()
        self.test_dataset.renew_data()

    def printLength(self):
        print(len(self.key_usable))
        print(len(self.key_trainable))


def main():
    dataset_handler = DatasetHandler()
    import gc
    gc.collect()
    print("Waiting")
    test = input()

if __name__ == "__main__":
    main()
