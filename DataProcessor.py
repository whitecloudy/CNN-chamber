import numpy as np
import torch
import random
import time
import cmath
import copy
import sys
import pickle

from Proc import do_work
from DataAugmentation import data_augmentation, aug_para
from DataHandler import DataHandler
from itertools import permutations
import ArgsHandler


class dataParser:
    def __init__(self, data, label, key):
        self.x_data = self.parse_data(data)
        self.y_data = self.parse_label(label)
        self.heur_data = self.cal_heuristic(data)
        self.key = key

    def cal_ls_estimator(self, W, A):
        if np.linalg.matrix_rank(W) < 6:
            raise NameError("Not enough rank")

        WHW = (W.getH() * W)

        W_inv = WHW.getI() * W.getH()
        H = (W_inv * A)
        H = H.getT().getA()[0]

        return H


    def cal_heuristic(self, data):
        W = []
        A = []

        for d in data:
            W.append(d.phase_vec)
            A.append([d.tag_sig])

        W = np.matrix(W)
        A = np.matrix(A)

        H = self.cal_ls_estimator(W, A)

        for i in range(6):
            label_element = (H[i])
            if cmath.isinf(label_element) or cmath.isnan(label_element):
                raise NameError("Inf or Nan for Heur")
        
        return np.array(H)
        
       
    def parse_data(self, data):
        x_data_list = np.array([d.x_row for d in data])
        
        return x_data_list


    def parse_label(self, label):
        return np.array(label)


global_data_handler = DataHandler()
global_key_list = list(global_data_handler.getKey())

import gc

class DataProcessor:
    def __init__(self, multiply=1, key_list=global_key_list, row_size=6, normalize=None):
        self.data_handler = global_data_handler
        self.output_list = []
        self.key_list = key_list
        self.data_label_key_list = []
        self.index_list = []
        self.index_len = 0
        self.multiply = 0
        self.row_size = row_size
        para_tuples = []

        # data augmentation
        for data, label, key in self.data_handler:
            if key not in self.key_list:
                continue

            if key[3] == ' directionalrefine':
                continue

            para_tuples.append((data, label, key))

        self.data_label_key_list = self.handle_cache(para_tuples, aug_para, data_augmentation)

        gc.collect()

        print("Data Augmentation Complete")

        # config normalize value
        if normalize is None:
            self.normalize = self.calculate_normalize()
        else:
            self.normalize = normalize
        
        # prepare output data
        self.prepare_data(multiply)


    def handle_cache(self, para_tuples, additional_tuples, func):
        import hashlib
        import os

        tuples_byte = pickle.dumps(para_tuples)
        add_byte = pickle.dumps(additional_tuples)
        hash_handler = hashlib.sha3_224()
        hash_handler.update(tuples_byte)
        hash_handler.update(add_byte)

        from pathlib import Path

        cache_filename = str(Path.home()) + "/cache/" + str(additional_tuples) + hash_handler.digest().hex()

        rt_data = []
        print(cache_filename)

        if os.path.isfile(cache_filename):
            print("Cache file found")
            with open(cache_filename, "rb") as cache_file:
                rt_data = pickle.load(cache_file)
        else:
            print("No Cache file found")
            for result in do_work(func, para_tuples, 16):
                rt_data += result
            
            with open(cache_filename, "wb") as cache_file:
                pickle.dump(rt_data, cache_file)
        
        return rt_data


    def prepare_data(self, multiply):
        self.output_list = []
        self.multiply = multiply
        para_tuples = []

        def make_output_data(data, label, key, row_size, multiply=1):
            prepared_data_list = []

            random.seed(time.time())

            last_idx = 0
            
            for count in range(multiply):
                idx = list(range(len(data)))
                random.shuffle(idx)

                for i in range(0, len(data) - row_size + 1, row_size):
                    data_to_parse = []
                    for j in range(i, i+row_size):
                        data_to_parse.append(data[idx[j]])

                    last_idx = i

                    parsedData = None

                    try:
                        parsedData = dataParser(data_to_parse, label, key)
                    except np.linalg.LinAlgError:
                        print("one go")
                        continue
                    except NameError:
                        continue

                    prepared_data_list.append(parsedData)

                # handle last remaining data
                if last_idx < len(data) - row_size:
                    data_to_parse = []
                    for j in range(len(data) - row_size, len(data)):
                        data_to_parse.append(data[idx[j]])

                    #data_to_parse = data[len(data) - row_size:len(data)]

                    parsedData = None

                    try:
                        parsedData = dataParser(data_to_parse, label, key)
                    except np.linalg.LinAlgError:
                        print("two go")
                        break
                    except NameError:
                        break

                    prepared_data_list.append(parsedData)
            return prepared_data_list

        for data, label, key in self.data_label_key_list:
            para_tuples.append((data, label, key, self.row_size, multiply))

        self.output_list = self.handle_cache(para_tuples, (self.multiply, self.row_size, "Nor"), make_output_data)

        gc.collect()

        print("Done")

        self.index_len = int(len(self.output_list)/self.multiply)
        self.index_list = list(range(len(self.output_list)))
        random.shuffle(self.index_list)

    def shuffle_data(self):
        print("Shuffle data")
        random.shuffle(self.index_list)

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

    def __len__(self):
        return self.index_len

    def __getitem__(self, idx):
        if self.index_len <= idx:
            raise IndexError
        else:
            idx = self.index_list[idx]

        x = torch.FloatTensor([self.output_list[idx].x_data.real, self.output_list[idx].x_data.imag])
        y = torch.FloatTensor([self.output_list[idx].y_data.real, self.output_list[idx].y_data.imag]).reshape(12,)
        heur = torch.FloatTensor([self.output_list[idx].heur_data.real, self.output_list[idx].heur_data.imag]).reshape(12,)

        return x, y, heur

def calculate_MMSE_parameter(datas):
    H_Y_pair = []
    Y_avg_sum = 0
    H_avg_sum = 0
    for data in datas.output_list:
        Y = data.heur_data
        H = data.y_data.reshape(6)

        H_Y_pair.append((H, Y))

        Y_avg_sum += Y

        H_avg_sum += H

    N = len(datas.output_list)

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


def main():
    datas = DataProcessor(multiply=1)

    # for x, y in d:
        #print(y.shape)

    #print(d.normalize)


if __name__ == "__main__":
    ArgsHandler.init_args()
    main()
