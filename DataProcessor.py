import numpy as np
import torch
import random
import time
import cmath
import copy
import sys
import pickle
import gc
import math

from Proc import do_work, pseudo_list
from DataAugmentation import data_augmentation, aug_para
from DataHandler import DataHandler
from itertools import permutations


global_data_handler = DataHandler()
global_key_list = list(global_data_handler.getKey())


class dataParser:
    def __init__(self, data, label, key):
        self.data = data
        self.label = label
        self.key = key


    def cal_ls_estimator(self, W, A):
        if np.linalg.matrix_rank(W) < 6:
            raise NameError("Not enough rank")

        WHW = (W.getH() * W)

        W_inv = WHW.getI() * W.getH()
        H = (W_inv * A)
        H = H.getT().getA()[0]

        return H


    def cal_heuristic(self):
        W = []
        A = []

        for d in self.data:
            phase_vec = d.phase_vec
            tag_sig = d.tag_sig
            W.append(phase_vec)
            A.append([tag_sig])

        W = np.matrix(W)
        A = np.matrix(A)
        

        H = self.cal_ls_estimator(W, A)

        for i in range(6):
            label_element = (H[i])
            if cmath.isinf(label_element) or cmath.isnan(label_element):
                raise NameError("Inf or Nan for Heur")
        
        return np.array(H)

    def check_rank(self):
        W = []
        for d in self.data:
            phase_vec = d.phase_vec
            W.append(phase_vec)

        if np.linalg.matrix_rank(W) < 6:
            return False
        else:
            return True

    def heur_data(self):
        return self.cal_heuristic()

    def x_data(self):
        return np.array([d.x_row for d in self.data])

    def y_data(self):
        return np.array(self.label)

    def key_data(self):
        return self.key


class DataProcessor:
    def __init__(self, multiply=1, key_list=global_key_list):
        self.data_handler = global_data_handler
        self.output_len_list = []
        self.output_idx_list = []
        self.key_list = key_list
        self.data_label_key_list = []
        self.index_list = []
        self.index_len = 0
        self.multiply = 0
        para_tuples = []


        # data augmentation
        for data, label, key in self.data_handler:
            if key[0:3] not in self.key_list:
                continue

            if key[3] == ' directionalrefine':
                continue

            para_tuples.append((data, label, key))
        
        cache_file_name = self.make_cache_hashname(para_tuples, aug_para)

        self.data_label_key_list = self.load_cache(cache_file_name)
        if self.data_label_key_list is None:
            self.data_label_key_list = do_work(data_augmentation, para_tuples, 16)
            self.save_cache(self.data_label_key_list, cache_file_name)

        gc.collect()

        print("Data Augmentation Complete")
        
        """
        # config normalize value
        if normalize is None:
            self.normalize = self.calculate_normalize()
        else:
            self.normalize = normalize
        """

        # prepare output data
        # self.data_list = self.prepare_data(multiply)

    def make_cache_hashname(self, para_tuples, additional_tuples):
        import hashlib
        tuples_byte = pickle.dumps(para_tuples)
        add_byte = pickle.dumps(additional_tuples)
        hash_handler = hashlib.sha3_224()
        hash_handler.update(tuples_byte)
        hash_handler.update(add_byte)

        cache_filename = str(additional_tuples) + hash_handler.digest().hex()

        return cache_filename

    def save_cache(self, save_data, cache_filename):
        import os
        from pathlib import Path
        cache_path = str(Path.home()) + "/cache/" + cache_filename

        with open(cache_path, "wb") as cache_file:
            pickle.dump(save_data, cache_file)


    def load_cache(self, cache_filename):
        import os

        from pathlib import Path
        
        cache_path = str(Path.home()) + "/cache/" + cache_filename

        rt_data = None
        print(cache_filename)
        if os.path.isfile(cache_path):
            print("Cache file found")
            with open(cache_path, "rb") as cache_file:
                rt_data = pickle.load(cache_file)
        else:
            print("No Cache file found")

        return rt_data


    def prepare_data(self, row_size, multiply):
        self.output_len_list = []
        self.output_idx_list = []
        self.multiply = multiply

        gc.collect()

        self.index_len = 0

        result_list = []
        for idx, dlk in enumerate(self.data_label_key_list):
            d = dlk[0]
            l = dlk[1]
            k = dlk[2]
            d_len = len(d)
            data_len = math.ceil(d_len/row_size)
            data_idx_list = list(range(d_len))

            for repeat in range(multiply):
                random.shuffle(data_idx_list)

                for j in range(data_len):
                    if (data_len-1) == j:
                        end_idx = d_len
                        start_idx = end_idx - row_size
                        x_data = [d[data_idx_list[i]] for i in range(start_idx, end_idx)]
                    else:
                        start_idx = j * row_size
                        end_idx = start_idx + row_size
                        x_data = [d[data_idx_list[i]] for i in range(start_idx, end_idx)]

                    data_c = dataParser(x_data, l, k)

                    if data_c.check_rank():
                        result_list.append((data_c.x_data(), data_c.heur_data(), data_c.y_data()))
                    else:
                        continue

        return result_list

    def calculate_normalize(self):
        mean_tag = 0.0
        mean_std = 0.0
        mean_label = np.zeros((6, 1))
        avg_tag = 0.0+0.0j
        avg_std = 0.0+0.0j
        avg_label = np.zeros((6, 1), dtype='complex128')

        data_len = 0

        for data, label, key in self.data_label_key_list:
            data_len += len(data)
            for d in data:
                mean_tag += abs(d.tag_sig)
                mean_std += abs(d.noise_std)
                avg_tag += d.tag_sig
                avg_std += d.noise_std
            mean_label += np.abs(label)
            avg_label += label

        mean_tag /= data_len
        mean_std /= data_len
        avg_tag /= data_len
        avg_std /= data_len

        mean_label /= len(self.data_label_key_list)
        avg_label /= len(self.data_label_key_list)

        return (mean_tag, mean_std, mean_label, avg_tag, avg_std, avg_label)


    def calculate_MMSE_parameter(self):
        H_Y_pair = []
        Y_avg_sum = 0
        H_avg_sum = 0

        for i in range(self.index_len):
            data = self.indexing_data(i)
            Y = data.heur_data()
            H = data.y_data().reshape(6)

            H_Y_pair.append((H, Y))

            Y_avg_sum += Y

            H_avg_sum += H

        N = self.index_len

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

        return r_hy * r_yy_inv

    """
    def __len__(self):
        return self.index_len


    def __getitem__(self, idx):
        if self.index_len <= idx:
            raise IndexError
        else:
            data = self.indexing_data(idx)

        heur_data = data.heur_data()
        x_data = data.x_data()
        y_data = data.y_data()

        x = torch.FloatTensor([x_data.real, x_data.imag])
        y = torch.FloatTensor([y_data.real, y_data.imag]).reshape(12,)
        heur = torch.FloatTensor([heur_data.real, heur_data.imag]).reshape(12,)

        return x, y, heur
    """


def main():
    key_list = global_key_list
    trainable_key_list = []
    error_thres = 0.15
    data_div = 10
    seed = 1

    for key in key_list:
        error, length = global_data_handler.evalLabel(key)
        if length >= 54:
            if error < error_thres:
                trainable_key_list.append(key[0:3])
    print(len(trainable_key_list))
    
    random.seed(seed)
    random.shuffle(trainable_key_list)

    key_len = len(trainable_key_list)
    key_step_len = int(key_len/data_div)
    key_remain = int(key_len - key_step_len*data_div)

    start_idx = 0
    end_idx = 0

    for i in range(data_div):
        start_idx = end_idx
        end_idx += key_step_len
        if key_remain > 0:
            key_remain -= 1
            end_idx += 1
        step_key = trainable_key_list[start_idx: end_idx]
        datas = DataProcessor(key_list=step_key)
        for row_size in range(6, 13):
            print("<<<", i, " ", row_size, ">>>")
            data_list = datas.prepare_data(multiply=1, row_size=row_size)
            filename = str(row_size)+"_"+str(i)+'_20211213.bin'
            datas.save_cache(data_list, filename)



if __name__ == "__main__":
    main()
