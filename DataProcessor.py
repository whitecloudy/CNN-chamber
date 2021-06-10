from DataHandler import DataHandler
import numpy as np
import torch
import random
import time

SIZE_OF_DATA = 27


class dataParser:
    def __init__(self, data, label, key):
        self.x_origin = data
        self.y_origin = label
        self.x_data = self.parse_data(data)
        self.y_data = self.parse_label(label)
        self.key = key
       
    def parse_data(self, data):
        real_list = []
        imag_list = []

        for i in range(SIZE_OF_DATA):
            real = []
            imag = []

            for phase in data[i].phase_vec:
                real.append(phase.real)
                imag.append(phase.imag)

            real.append(data[i].tag_sig.real)
            imag.append(data[i].tag_sig.imag)

            real.append(data[i].noise_std.real)
            imag.append(data[i].noise_std.imag)

            real_list.append(real)
            imag_list.append(imag)

        return torch.FloatTensor(np.array([real_list, imag_list]))

    def parse_label(self, label):
        y_data = []

        for i in range(6):
            y_data.append(label[i].real)
            y_data.append(label[i].imag)

        return torch.FloatTensor(np.array(y_data))


global_data_handler = DataHandler()
global_key_list = list(global_data_handler.getKey())


class DataProcessor:
    def __init__(self, multiply=1, key_list=global_key_list):
        self.data_handler = global_data_handler
        self.data_label_list = []
        self.data_key_list = key_list

        for data, label, key in self.data_handler:
            if key not in self.data_key_list:
                continue
            self.data_label_list += self.prepare_data(data=data,
                                                      label=label,
                                                      key=key,
                                                      multiply=multiply)

    def prepare_data(self, data, label, key, multiply=1):
        prepared_data_list = []

        random.seed(time.time())

        last_idx = 0
        
        for count in range(multiply):
            random.shuffle(data)

            for i in range(0, len(data) - SIZE_OF_DATA + 1, SIZE_OF_DATA):
                data_to_parse = data[i:i+SIZE_OF_DATA]

                prepared_data_list.append(dataParser(data_to_parse, label, key))

                last_idx = i

            # handle last remaining data
            if last_idx < len(data) - SIZE_OF_DATA:
                data_to_parse = data[len(data) - SIZE_OF_DATA:len(data)]

                prepared_data_list.append(dataParser(data_to_parse, label, key))

        return prepared_data_list

    def __len__(self):
        return len(self.data_label_list)

    def __getitem__(self, idx):
        return self.data_label_list[idx].x_data, self.data_label_list[idx].y_data


def main():
    d = DataProcessor(multiply=5)

    print(len(d))


if __name__ == "__main__":
    main()
