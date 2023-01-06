import random
import time
import torch
import numpy as np
import ArgsHandler
import Proc
import cmath
from CachefileHandler import load_cache, save_cache, make_cache_hashname

from torch.utils.data import Dataset

data_filename = []
data_segments = []

total_div_len = 40
dry_run_len = 10

def prepare_dataset(row_size, multiply, dry_run=False):
    for i in range(total_div_len):
        if dry_run:
            if i == dry_run_len:
                break

        filename = str(row_size)+"_"+str(multiply)+"_"+str(i)+'_20220325_ver114.bin'

        loaded_data = load_cache(filename)

        # data will be loaded in (x, h, y)
        data_filename.append(filename)
        data_segments.append(loaded_data)
        print(len(data_filename))

class BeamDataset(Dataset):
    def __init__(self, multiply, num_list, data_size=6, normalize=None, MMSE_para=None, aug_para=None):
        self.multiply = multiply
        self.data_size = data_size
        self.aug_para = aug_para
        self.aug_multiply = 1
        
        print(len(data_filename))
        cache_filename_list = [(data_filename[i], ) for i in num_list]
        self.hashname = make_cache_hashname(cache_filename_list)
        self.data_list = [np.empty((0, self.data_size, 8)), np.empty((0, self.data_size)), np.empty((0, self.data_size, 1))]
        #self.len_list = []
        self.total_len = 0

        for i, idx in enumerate(num_list):
            # data will be loaded in (x, h, y)
            # self.data_list[0] = np.append(self.data_list[0], data_segments[idx][0], axis=0)
            # self.data_list[1] = np.append(self.data_list[1], data_segments[idx][1], axis=0)
            # self.data_list[2] = np.append(self.data_list[2], data_segments[idx][2], axis=0)
            #self.len_list.append(len(data_segments[idx][0]))
            #self.total_len += self.len_list[i]
            self.total_len += len(data_segments[idx][0])
            #print(self.len_list[i])
            print(self.total_len)

        self.data_list[0] = np.concatenate([data_segments[i][0] for i in num_list], axis=0)
        self.data_list[1] = np.concatenate([data_segments[i][1] for i in num_list], axis=0)
        self.data_list[2] = np.concatenate([data_segments[i][2] for i in num_list], axis=0)

        if aug_para is None:
            self.aug_multiply = 1
        else:
            for aug in aug_para:
                self.aug_multiply *= aug
        
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
        C_h = 0
        #C_w_elem = 0
        C_w = 0

        for x_mat_list, h_list, ls_list in self.data_list:
            x_split = np.split(x_mat_list, (6, 7, 8), axis=2)
            S_array = x_split[0]
            y_array = x_split[1]
            w_array = x_split[2]
            #w_array = np.split(x_mat_list, (7, ), axis=2)[1]
            h_array = h_list.reshape((len(h_list), 6, 1))

            w_array = np.matmul(S_array, h_array) - y_array

            w_H_array = np.conj(np.transpose(w_array, axes=(0, 2, 1)))
            h_H_array = np.conj(np.transpose(h_array, axes=(0, 2, 1)))
            h_h_array = np.matmul(h_array, h_H_array)
            #print(h_h_array)
            C_h += np.sum(h_h_array, axis=0)
            #C_w_elem += (np.sum(abs(w_array)/self.data_size))
            #C_w_elem += np.sum(np.matmul(w_array, w_H_array), axis=0)
            C_w += np.sum(np.matmul(w_array, w_H_array), axis=0)


        N = self.total_len

        C_h /= N
        #C_w_elem /= N
        C_w /= N

        #C_w = np.zeros((self.data_size, self.data_size), complex)
        #np.fill_diagonal(C_w, C_w_elem)

        return C_h, C_w


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
                C_h, C_w = self.calculate_MMSE_parameter()
                self.MMSE_para = (torch.tensor(C_h, dtype=torch.complex128).to("cpu"), torch.tensor(C_w, dtype=torch.complex128).to("cpu"))
                save_cache(self.MMSE_para, cache_name)

        return self.MMSE_para

    
    def do_aug(self, x, h, y):
        x_aug_vector = np.ones(8, dtype=np.complex128)
        h_aug_vector = np.ones(6, dtype=np.complex128)
        y_aug_vector = np.ones((6,1), dtype=np.complex128)

        if self.aug_para[0] != 1:
            rand_val = np.random.randint(self.aug_para[0])
            #zero_do_rand_val = np.random.randint(2)
            zero_do_rand_val = 0
            if rand_val != 0 or zero_do_rand_val != 0:
                fix_shift = cmath.rect(1, 2*cmath.pi*(rand_val/self.aug_para[0])) 
                random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()/self.aug_para[0]))

                shift_val = fix_shift * random_shift
             
                x_aug_vector[6:8] *= shift_val
                h_aug_vector *= shift_val
                y_aug_vector *= shift_val
        
        if self.aug_para[1] != 1:
            rand_val = np.random.randint(self.aug_para[1])
            zero_do_rand_val = np.random.randint(2)
            if rand_val != 0 or zero_do_rand_val != 0:
                fix_shift = cmath.rect(1, 2*cmath.pi*(rand_val/self.aug_para[1])) 
                random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()/self.aug_para[1]))

                shift_val = fix_shift * random_shift

                x_aug_vector[0:6] *= shift_val
                h_aug_vector *= shift_val.conjugate()
                y_aug_vector *= shift_val.conjugate()
        
        x *= x_aug_vector
        d = np.split(x, [7,], axis=1)
        d[1] = abs(d[1].real) + abs(d[1].imag)*1j
        x = np.append(d[0], d[1], axis=1)
        y *= y_aug_vector
        h *= h_aug_vector

        return x, h, y


    def get_data(self, idx):
        # i = 0
        # j = 0
        # while True:
        #     length = self.len_list[i]
        #     if idx < length: 
        #         j = idx
        #         break
        #     else:
        #         idx -= length
        #         i += 1

        # x = self.data_list[i][0][j]
        # h = self.data_list[i][1][j]
        # y = self.data_list[i][2][j]

        x = self.data_list[0][idx]
        h = self.data_list[1][idx]
        y = self.data_list[2][idx]

        if self.aug_para is not None:
            x, h, y = self.do_aug(x, h, y)

        return x, h, y


    def __len__(self):
        return int(self.total_len * self.aug_multiply)

    def __getitem__(self, idx):
        x, h, y = self.get_data(int(idx/self.aug_multiply))

        x = torch.FloatTensor(np.append(np.expand_dims(x.real, axis=0), np.expand_dims(x.imag, axis=0), axis=0))
        y = torch.FloatTensor(np.append(y.real, y.imag)).reshape(12,)
        h = torch.FloatTensor(np.append(h.real, h.imag)).reshape(12,)

        return x, h, y


class DatasetHandler:
    def __init__(self,  multiply=1, data_div=5, val_data_num=1, row_size=6, aug_para=None):
        self.multiply = multiply
        self.data_div = data_div
        self.val_data_num = val_data_num
        self.row_size = row_size

        self.training_dataset = None
        self.training_test_dataset = None
        self.test_dataset = None
        self.aug_para = aug_para
        
        self.prepare_dataset()

    def prepare_dataset(self):
        data_div = self.data_div
        val_data_num = self.val_data_num

        nums_for_training = []
        nums_for_validation = []

        data_div_len = len(data_filename)

        for i in range(data_div):
            step_num_list = list(range(int(i * data_div_len/data_div), int((i+1) * data_div_len/data_div)))

            if i == val_data_num:
                nums_for_validation += step_num_list
            else:
                nums_for_training += step_num_list
                
        self.training_dataset = BeamDataset(self.multiply, nums_for_training, self.row_size, aug_para=self.aug_para)#, self.normalize)
        self.normalize = self.training_dataset.getNormPara()
        self.training_test_dataset = BeamDataset(self.multiply, nums_for_training, self.row_size, self.normalize)
        self.test_dataset = BeamDataset(self.multiply, nums_for_validation, self.row_size, self.normalize)


def main():
    prepare_dataset(12, 1)
    dataset_handler = DatasetHandler(data_div=5, val_data_num=1, row_size=12)
    dataset_handler.training_dataset.getMMSEpara()

    import gc
    gc.collect()
    print("Waiting")
    test = input()

if __name__ == "__main__":
    main()
