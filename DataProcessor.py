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
from CachefileHandler import save_cache, load_cache


global_data_handler = DataHandler(True)
global_key_list = list(global_data_handler.getKey())
global_validation_data_handler = DataHandler(False)
global_validation_key_list = list(global_validation_data_handler.getKey())


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
    def __init__(self, data_handler, key_list, multiply=1):
        self.data_handler = data_handler
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

        self.data_label_key_list = load_cache(cache_file_name)
        if self.data_label_key_list is None:
            self.data_label_key_list = do_work(data_augmentation, para_tuples, 16)
            save_cache(self.data_label_key_list, cache_file_name)

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


    def prepare_data(self, row_size, multiply):
        self.output_len_list = []
        self.output_idx_list = []
        self.multiply = multiply

        gc.collect()

        self.index_len = 0
        
        x_result = []#np.empty((0, row_size, 8))
        y_result = []#np.empty((0, 6, 1))
        h_result = []#np.empty((0, 6))
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
                        try:
                            x_result.append(data_c.x_data())
                            y_result.append(data_c.y_data())
                            h_result.append(data_c.heur_data())
                        except np.linalg.LinAlgError:
                            print("LinAlg Error")
                            continue
                        except NameError:
                            print("Name Error")
                            continue
                    else:
                        continue
        x_result = np.array(x_result)
        h_result = np.array(h_result)
        y_result = np.array(y_result)

        return x_result, h_result, y_result


def work_for_preparing(data_c, i, multiply, row_size, prefix=''):
    print("<<<", i, " ", row_size, ">>>")
    data_list = data_c.prepare_data(multiply=multiply, row_size=row_size)
    filename = prefix + str(row_size)+"_"+str(multiply)+"_"+str(i)+'_20220325_ver111.bin'
    save_cache(data_list, filename)
    print("Done ", i, " ", row_size)

    return [0, ]


def make_data(data_handler : DataHandler, key_list : list, prefix : str, error_thres : float, data_div=40):
    trainable_key_list = []
    seed = 1
    multiply = 1

    for key in key_list:
        try:
            error, length = data_handler.evalLabel(key)
        except KeyError:
            print(key, " : It is not Usable")
            continue
        if length >= 250:
            if error < error_thres:
                trainable_key_list.append(key[0:3])
            else:
                print(key, " is Too much Error : ", error)
        else:
            print(key, " has not enough data")
    print(len(trainable_key_list))
    
    random.seed(seed)
    random.shuffle(trainable_key_list)

    key_len = len(trainable_key_list)
    key_step_len = int(key_len/data_div)
    key_remain = int(key_len - key_step_len*data_div)

    start_idx = 0
    end_idx = 0

    for i in range(0, data_div, 2):
        print("\nNow working!!")
        print(i)
        print()
        start_idx = end_idx
        end_idx += key_step_len
        if key_remain > 0:
            key_remain -= 1
            end_idx += 1
        step_key1 = trainable_key_list[start_idx: end_idx]

        start_idx = end_idx
        end_idx += key_step_len
        if key_remain > 0:
            key_remain -= 1
            end_idx += 1
        step_key2 = trainable_key_list[start_idx: end_idx]

        if True:
            data1 = DataProcessor(data_handler, key_list=step_key1)
            data2 = DataProcessor(data_handler, key_list=step_key2)

            para_tuples = []
            for row_size in range(6, 13):
                para_tuples.append((data1, i, multiply, row_size, prefix))

            for row_size in range(6, 13):
                para_tuples.append((data2, i+1, multiply, row_size, prefix))

            do_work(work_for_preparing, para_tuples, 16)

def main():
    make_data(global_data_handler, global_key_list, "training_", 0.4)

    make_data(global_validation_data_handler, global_validation_key_list, "validation_", 1.0)
    


if __name__ == "__main__":
    main()
