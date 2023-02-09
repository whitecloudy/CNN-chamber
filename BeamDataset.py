import random
import time
import torch
import numpy as np
import ArgsHandler
import Proc
import cmath
from CachefileHandler import load_cache, save_cache, make_cache_hashname

from torch.utils.data import Dataset



total_div_len = 40
dry_run_len = 10

shuffle_candidate = np.array([[0, 1, 2, 3, 4, 5], [2, 1, 0, 5, 4, 3], [3, 4, 5, 0, 1, 2], [5, 4, 3, 2, 1, 0]])


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

def prepare_dataset(row_size, multiply, dry_run=False):
    training_data_filename = []
    training_data_segments = []
    validation_data_filename = []
    validation_data_segments = []
    
    for i in range(total_div_len):
        if dry_run:
            if i == dry_run_len:
                break

        filename = 'training_'+str(row_size)+"_"+str(multiply)+"_"+str(i)+'_20220325_ver111.bin'

        loaded_data = load_cache(filename)

        # data will be loaded in (x, h, y)
        training_data_filename.append(filename)
        training_data_segments.append(loaded_data)

        filename = 'validation_'+str(row_size)+"_"+str(multiply)+"_"+str(i)+'_20220325_ver111.bin'

        loaded_data = load_cache(filename)

        # data will be loaded in (x, h, y)
        validation_data_filename.append(filename)
        validation_data_segments.append(loaded_data)
        print(len(validation_data_filename))

    

class BeamDataset(Dataset):
    def __init__(self, data_filename_list : list[str], data_segments=None, data_size=6, normalize=None, MMSE_para=None, aug_ratio=None):
        self.data_size = data_size
        
        print(len(data_filename_list))
        self.hashname = make_cache_hashname(data_filename_list)
        self.x_list = np.empty((0, self.data_size, 8))
        self.h_list = np.empty((0, self.data_size))
        self.y_list = np.empty((0, self.data_size, 1))
        #self.len_list = []
        self.total_len = 0

        if data_segments is None:
            self.load_data_segments(data_filename_list)
        else:
            self.x_list = data_segments[0]
            self.h_list = data_segments[1]
            self.y_list = data_segments[2]

        self.total_len = len(self.x_list)

        self.MMSE_para = MMSE_para
        self.normalize = normalize
        self.aug_ratio = aug_ratio

    def get_data_segments(self) -> tuple:
        return (self.x_list, self.h_list, self.y_list)
    
    def load_data_segments(self, filename_list : list[str]):
        data_segments = [load_cache(filename) for filename in filename_list]

        self.x_list = np.concatenate([data_segments[i][0] for i in range(len(data_segments))], axis=0)
        self.h_list = np.concatenate([data_segments[i][1] for i in range(len(data_segments))], axis=0)
        self.y_list = np.concatenate([data_segments[i][2] for i in range(len(data_segments))], axis=0)


    def calculate_normalize(self):
        x_mean = 0
        y_mean = 0
        h_mean = 0

        x_mean += np.sum(np.sum(abs(np.split(self.x_list, (6, ), axis=2)[1]), axis=0), axis=0)
        y_mean += np.sum(abs(self.y_list), axis=0).reshape((1, 6))
        h_mean += np.sum(abs(self.h_list), axis=0).reshape((1, 6))

        x_mean /= (self.total_len * self.data_size)
        x_mean = np.append([1. for i in range(6)], x_mean)
        y_mean /= self.total_len
        h_mean /= self.total_len

        return (1/x_mean, 1/h_mean, 1/y_mean)
 

    def calculate_MMSE_parameter(self):
        C_h = 0
        #C_w_elem = 0
        C_w = 0

        x_mat_list = self.x_list
        h_list = self.h_list

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

        if np.random.rand() < self.aug_ratio:
            shift_val_1 = cmath.rect(1, 2*cmath.pi*(np.random.rand()))
            
            shift_val_2 = cmath.rect(1, 2*cmath.pi*(np.random.rand()))

            x_aug_vector[6:8] *= shift_val_1
            x_aug_vector[0:6] *= shift_val_2
            h_aug_vector *= (shift_val_2.conjugate() * shift_val_1)
            y_aug_vector *= (shift_val_2.conjugate() * shift_val_1)

            shuffle_order = shuffle_candidate[np.random.randint(4)]
       
            x *= x_aug_vector
            x[:,[7]] = abs(x[:,[7]].real) + abs(x[:,[7]].imag)*1j
            x[:,[0,1,2,3,4,5]] = x[:,shuffle_order]

            y *= y_aug_vector
            y = y[shuffle_order]

            h *= h_aug_vector
            h = h[shuffle_order]

        return x, h, y

    def __len__(self):
        return int(self.total_len)

    def __getitem__(self, idx) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        x = self.x_list[idx]
        h = self.h_list[idx]
        y = self.y_list[idx]
        
        if self.aug_ratio != None:
            x, h, y = self.do_aug(x, h, y)

        x = torch.FloatTensor(np.append(np.expand_dims(x.real, axis=0), np.expand_dims(x.imag, axis=0), axis=0))
        y = torch.FloatTensor(np.append(y.real, y.imag)).reshape(12,)
        h = torch.FloatTensor(np.append(h.real, h.imag)).reshape(12,)

        # x = torch.tensor(x, dtype=torch.cdouble)
        # h = torch.tensor(h, dtype=torch.cdouble).reshape(6,)
        # y = torch.tensor(y, dtype=torch.cdouble).reshape(6,)

        return x, h, y


class DatasetHandler:
    def __init__(self,  multiply=1, row_size=6, aug_ratio=None, dry_run=False):
        self.multiply = multiply
        self.row_size = row_size

        self.training_dataset = None
        self.training_test_dataset = None
        self.validation_dataset = None

        self.aug_ratio = aug_ratio
        
        self.prepare_dataset(dry_run)

    def prepare_dataset(self, dry_run=False):
        if dry_run:
            total_div_len = 10

        training_filename_list = ['training_'+str(self.row_size)+"_"+str(self.multiply)+"_"+str(i)+'_20220325_ver111.bin' for i in range(total_div_len)]
        validation_filename_list = ['validation_'+str(self.row_size)+"_"+str(self.multiply)+"_"+str(i)+'_20220325_ver111.bin' for i in range(total_div_len)]
             
        self.training_dataset = BeamDataset(training_filename_list, data_size=self.row_size, aug_ratio=self.aug_ratio)#, self.normalize)
        self.normalize = self.training_dataset.getNormPara()
        self.validation_dataset = BeamDataset(validation_filename_list, data_size=self.row_size, normalize=self.normalize)
        self.training_test_dataset = BeamDataset(training_filename_list, self.training_dataset.get_data_segments(), self.row_size, self.normalize)


def main():
    dataset_handler = DatasetHandler(row_size=12, dry_run=True)
    test_d = dataset_handler.validation_dataset

    test_kwargs = {'batch_size': 256}

    loader = torch.utils.data.DataLoader(test_d, **test_kwargs)

    for x, h, y in loader:
        print(x.shape)
        random = torch.rand(256, 12, 8, dtype=torch.complex128)
        random = random/random.abs()
        print(random.abs())
        #print(random)
        break

    import gc
    gc.collect()
    print("Waiting")
    test = input()

if __name__ == "__main__":
    main()
