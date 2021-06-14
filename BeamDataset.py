import torch
from numpy.random import randn
from torch.utils.data import Dataset
from DataProcessor import DataProcessor, global_key_list, global_data_handler
import random
import time

class BeamDataset(Dataset):
    def __init__(self, multiply, key_list, normalize=None):
        self.data_processor = DataProcessor(multiply, key_list, normalize)

    def __len__(self):
        return len(self.data_processor)

    def __getitem__(self, idx):
        return self.data_processor[idx]

class DatasetHandler:
    def __init__(self, error_thres=0.15, train_data_ratio=0.9, multiply=10):
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
        random.seed(time.time())
        random.shuffle(self.key_trainable)

        training_key_length = int(len(self.key_trainable) * self.train_data_ratio)
        
        for key in self.key_trainable:
            if key[0] == 'data/normal_210519.csv':
                self.key_for_test.append(key)
            else:
                self.key_for_training.append(key)

        self.key_for_test += self.key_usable
        #self.key_for_training = self.key_trainable[0:training_key_length]
        #self.key_for_test = self.key_trainable[training_key_length:] + self.key_usable

        self.training_dataset = BeamDataset(self.multiply, self.key_for_training)
        self.normalize = self.training_dataset.data_processor.normalize
        self.test_dataset = BeamDataset(self.multiply, self.key_for_test, self.normalize)


    def printLength(self):
        print(len(self.key_usable))
        print(len(self.key_trainable))


def main():
    d = DatasetHandler(error_thres=0.15)
    d.printLength()


if __name__ == "__main__":
    main()
