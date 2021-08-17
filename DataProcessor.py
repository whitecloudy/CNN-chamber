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

SIZE_OF_DATA = 27


class dataParser:
    def __init__(self, data, label, key, normalize):
        self.x_origin = data
        self.y_origin = label
        self.tag_norm = normalize[0]
        self.noise_norm = normalize[1]
        self.channel_norm = normalize[2]
        self.x_data = self.parse_data(data)
        self.y_data = self.parse_label(label)
        self.heur_data = self.cal_heuristic(data)
        self.key = key


    def cal_heuristic(self, data):
        W = []
        A = []

        W_size = ArgsHandler.args.heu

        if W_size < 6:
            return self.y_data

        for d in data:
            W.append(d.phase_vec)
            A.append([d.tag_sig])
            if W_size <= 1:
                break
            else:
                W_size -= 1

        W = np.matrix(W)
        A = np.matrix(A)

        try:
            WHW = (W.getH() * W)

            W_inv = WHW.getI() * W.getH()
            H = (W_inv * A)
            H = H.getT().getA()[0]

            result_data = []

            for i in range(6):
                label_element = (H[i]/abs(H[i]))
                norm = float(self.channel_norm[i])
                result_data.append(label_element.real/norm)

            for i in range(6):
                label_element = (H[i]/abs(H[i]))
                norm = float(self.channel_norm[i])
                result_data.append(label_element.imag/norm)
            return np.array(result_data)

        except np.linalg.LinAlgError:
            for i in range(6):
                result_data.append(float('inf'))
                result_data.append(float('inf'))
            return result_data

       
    def parse_data(self, data):
        real_list = []
        imag_list = []

        for i in range(SIZE_OF_DATA):
            real = []
            imag = []

            for phase in data[i].phase_vec:
                real.append(phase.real)
                imag.append(phase.imag)

            real.append(data[i].tag_sig.real/self.tag_norm)
            imag.append(data[i].tag_sig.imag/self.tag_norm)

            real.append(data[i].noise_std.real/self.noise_norm)
            imag.append(data[i].noise_std.imag/self.noise_norm)

            real_list.append(real)
            imag_list.append(imag)
        return_val = np.array([real_list, imag_list])

        return return_val


    def parse_label(self, label):
        y_data = []

        for i in range(6):
            label_element = complex(label[i])
            norm = float(self.channel_norm[i])
            y_data.append(label_element.real/norm)

        for i in range(6):
            label_element = complex(label[i])
            norm = float(self.channel_norm[i])
            y_data.append(label_element.imag/norm)

        return np.array(y_data)


global_data_handler = DataHandler()
global_key_list = list(global_data_handler.getKey())

import gc

class DataProcessor:
    def __init__(self, multiply=1, key_list=global_key_list, normalize=None):
        self.data_handler = global_data_handler
        self.output_list = []
        self.key_list = key_list
        self.data_label_key_list = []
        self.index_list = []
        self.index_len = 0
        self.multiply = 0
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
        cache_filename = "cache/" + hash_handler.digest().hex()

        rt_data = []

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

        def make_output_data(data, label, key, normalize, multiply=1):
            prepared_data_list = []

            random.seed(time.time())

            last_idx = 0
            
            for count in range(multiply):
                random.shuffle(data)

                for i in range(0, len(data) - SIZE_OF_DATA + 1, SIZE_OF_DATA):
                    data_to_parse = data[i:i+SIZE_OF_DATA]

                    prepared_data_list.append(dataParser(data_to_parse, label, key, normalize))

                    last_idx = i

                # handle last remaining data
                if last_idx < len(data) - SIZE_OF_DATA:
                    data_to_parse = data[len(data) - SIZE_OF_DATA:len(data)]

                    prepared_data_list.append(dataParser(data_to_parse, label, key, normalize))

            return prepared_data_list

        for data, label, key in self.data_label_key_list:
            para_tuples.append((data, label, key, self.normalize, multiply))

        self.output_list = self.handle_cache(para_tuples, (self.multiply), make_output_data)

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

        return torch.FloatTensor(self.output_list[idx].x_data), torch.FloatTensor(self.output_list[idx].y_data), torch.FloatTensor(self.output_list[idx].heur_data)


def main():
    d = DataProcessor(multiply=4)

    # for x, y in d:
        #print(y.shape)

    print(len(d))
    print(d[0])

    #print(d.normalize)


if __name__ == "__main__":
    ArgsHandler.init_args()
    main()
