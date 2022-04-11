import random
import time
import torch
import numpy as np
import ArgsHandler
import Proc
from CachefileHandler import load_cache, save_cache, make_cache_hashname

from torch.utils.data import Dataset

data_filename = []
data_segments = []

total_div_len = 40

def prepare_dataset(row_size, multiply):
    for i in range(total_div_len):
        filename = str(row_size)+"_"+str(multiply)+"_"+str(i)+'_20220325.bin'

        loaded_data = load_cache(filename)

        # data will be loaded in (x, h, y)
        data_filename.append(filename)
        data_segments.append(loaded_data)
        print(len(data_filename))
        if i==4:
            break

class BeamDataset(Dataset):
    def __init__(self, multiply, num_list, data_size=6, normalize=None, MMSE_para=None):
        self.multiply = multiply
        self.data_size = data_size
        
        print(len(data_filename))
        cache_filename_list = [(data_filename[i], ) for i in num_list]
        self.hashname = make_cache_hashname(cache_filename_list)
        self.data_list = []
        self.len_list = []
        self.total_len = 0

        for i, idx in enumerate(num_list):
            # data will be loaded in (x, h, y)
            self.data_list.append(data_segments[idx])
            self.len_list.append(len(data_segments[idx][0]))
            self.total_len += self.len_list[i]
            print(self.len_list[i])

        self.MMSE_para = MMSE_para
        self.normalize = normalize

    def calculate_normalize(self):
        x_mean = 0
        y_mean = 0
        h_mean = 0
        for x_mat, h, y in self.data_list:
            x_mean += np.sum(np.sum(abs(np.split(x_mat, (6, ), axis=2)[1]), axis=0), axis=0)
            y_mean += np.sum(abs(y), axis=0).reshape((1, 6))
            h_mean += np.sum(abs(h), axis=0).reshape((1, 6))

        x_mean /= (self.total_len * self.data_size)
        x_mean = np.append([1. for i in range(6)], x_mean)
        y_mean /= self.total_len
        h_mean /= self.total_len

        return (1/x_mean, 1/h_mean, 1/y_mean)
 

    def calculate_MMSE_parameter(self):
        Y_avg_sum = 0
        H_avg_sum = 0
        
        for x_mat_list, h_list, y_list in self.data_list:
            Y_avg_sum += np.sum(h_list, axis=0).reshape((1, 6))
            H_avg_sum += np.sum(y_list, axis=0).reshape((1, 6))

        N = self.total_len

        mu_y = Y_avg_sum / N
        mu_h = H_avg_sum / N

        r_hy = 0
        r_yy = 0

        for x_mat_list, h_list, y_list in self.data_list:
            H_hat_array = y_list.reshape((len(y_list), 6)) - mu_h
            Y_hat_array = h_list.reshape((len(h_list), 6)) - mu_y

            r_hy_array = H_hat_array.reshape((len(x_mat_list), 6, 1)) * np.conj(Y_hat_array.reshape((len(x_mat_list), 1, 6)))
            r_yy_array = Y_hat_array.reshape((len(x_mat_list), 6, 1)) * np.conj(Y_hat_array.reshape((len(x_mat_list), 1, 6)))

            r_hy += np.sum(r_hy_array, axis=0)
            r_yy += np.sum(r_yy_array, axis=0)
            
        r_hy /= N
        r_yy /= N

        r_yy_inv = np.matrix(r_yy).getI()
        mmse = r_hy * r_yy_inv

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
        return int(self.total_len)

    def __getitem__(self, idx):
        i = 0
        j = 0
        while True:
            length = self.len_list[i]
            if idx < length: 
                j = idx
                break
            else:
                idx -= length
                i += 1

        x = self.data_list[i][0][j]
        h = self.data_list[i][1][j]
        y = self.data_list[i][2][j]

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

        """
        for i in range(data_div):
            step_num_list = list(range(int(i * total_div_len/data_div), int((i+1) * total_div_len/data_div)))

            if i == val_data_num:
                nums_for_validation += step_num_list
            else:
                nums_for_training += step_num_list
        """
        nums_for_validation += [0, 1]
        nums_for_training += [2, 3, 4]
        
        self.training_dataset = BeamDataset(self.multiply, nums_for_training, self.row_size)#, self.normalize)
        self.normalize = self.training_dataset.getNormPara()
        self.test_dataset = BeamDataset(self.multiply, nums_for_validation, self.row_size, self.normalize)


def main():
    dataset_handler = DatasetHandler()
    import gc
    gc.collect()
    print("Waiting")
    test = input()

if __name__ == "__main__":
    main()
