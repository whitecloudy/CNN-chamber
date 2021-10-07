import random
import time
import torch
import numpy as np
import ArgsHandler

from torch.utils.data import Dataset
from DataProcessor import DataProcessor, global_key_list, global_data_handler, calculate_MMSE_parameter

class BeamDataset(Dataset):
    def __init__(self, multiply, key_list, normalize=None):
        self.data_processor = DataProcessor(multiply, key_list, normalize)
        self.MMSE_para = calculate_MMSE_parameter(self.data_processor)

    def renew_data(self, multiply):
        self.data_processor.shuffle_data()

    def getMMSEparaa(self):
        return self.MMSE_para

    def __len__(self):
        return len(self.data_processor)

    def __getitem__(self, idx):
        return self.data_processor[idx]

class DatasetHandler:
    def __init__(self, error_thres=0.15, train_data_ratio=0.9, multiply=5):
        self.key_list = global_key_list
        self.key_usable = []
        self.key_trainable = []
        self.key_for_training = []
        self.key_for_test = []

        self.multiply = multiply
        self.train_data_ratio = train_data_ratio

        self.training_dataset = None
        self.test_dataset = None

        for key in self.key_list:
            error, length = global_data_handler.evalLabel(key)
            if length >= 54:
                if error < error_thres:
                    self.key_trainable.append(key)
                else:
                    self.key_usable.append(key)
                    print("Data ", key, " will be ignored.")
                    print("Error : {:.4f}".format(error), "\tLength : ", length)
                    print()
        print(len(self.key_usable)+len(self.key_trainable))

        self.prepare_dataset()

    def prepare_dataset(self):
        random.seed(ArgsHandler.args.seed)
        random.shuffle(self.key_trainable)

        training_key_length = int(len(self.key_trainable) * self.train_data_ratio)
        
        key_position = []
        for key in self.key_trainable:
            pos = key[0:3]
            if pos not in key_position:
                key_position.append(pos)
            """
            if key[0] == 'data/normal_210519.csv':
                self.key_for_test.append(key)
            else:
                self.key_for_training.append(key)
            """
        random.shuffle(key_position)
        
        data_div = ArgsHandler.args.data_div
        val_data_num = ArgsHandler.args.val_data_num
        
        key_len = len(key_position)
        key_step_len = int(key_len/data_div)
        key_remain = int(key_len - key_step_len*data_div)
        
        key_range = []

        s_idx = 0
        e_idx = key_step_len

        if key_remain > 0:
            e_idx += 1

        for i in range(data_div):
            key_range.append((s_idx, e_idx))
            print(e_idx)
            
            s_idx = e_idx
            e_idx = e_idx + key_step_len

            if i+1 < key_remain:
                e_idx += 1

        training_pos = []
        test_pos = []

        for idx, rng in enumerate(key_range):
            if idx == val_data_num:
                test_pos = key_position[rng[0]: rng[1]]
            else:
                training_pos += key_position[rng[0]: rng[1]]

        for key in self.key_trainable:
            if key[0:3] in training_pos:
                self.key_for_training.append(key)
            elif key[0:3] in test_pos:
                self.key_for_test.append(key)

        self.key_for_test += self.key_usable
        #self.key_for_training = self.key_trainable[0:training_key_length]
        #self.key_for_test = self.key_trainable[training_key_length:] + self.key_usable
        #self.normalize = (0.028442474880584625, 0.0002280199237627333, np.array([0.02179932, 0.03584705, 0.02130222, 0.00743575, 0.00666348, 0.00799966]))
        
        self.training_dataset = BeamDataset(self.multiply, self.key_for_training)#, self.normalize)
        self.normalize = self.training_dataset.data_processor.normalize
        self.training_normalize = self.training_dataset.data_processor.normalize

        self.test_dataset = BeamDataset(self.multiply, self.key_for_test, self.normalize)
        self.testing_normalize = self.test_dataset.data_processor.normalize


    def renew_dataset(self):
        self.training_dataset.renew_data(self.multiply)
        self.test_dataset.renew_data(self.multiply)


    def printLength(self):
        print(len(self.key_usable))
        print(len(self.key_trainable))


def main():
    d = DatasetHandler(error_thres=0.15)
    print("<<<<<<<Training>>>>>>>>")
    for norm in d.training_normalize:
        print(abs(norm))
        print()
    print()
    print("<<<<<<<Validation>>>>>>>>")
    for norm in d.testing_normalize:
        print(abs(norm))
        print()
    
    d.printLength()


if __name__ == "__main__":
    main()
