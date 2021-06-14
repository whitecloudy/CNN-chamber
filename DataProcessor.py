from DataHandler import DataHandler
import numpy as np
import torch
import random
import time

SIZE_OF_DATA = 27


class dataParser:
    def __init__(self, data, label, key, normalize):
        self.x_origin = data
        self.y_origin = label
        self.tag_norm = normalize[0]
        self.noise_norm = normalize[1]
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

            real.append(data[i].tag_sig.real/self.tag_norm)
            imag.append(data[i].tag_sig.imag/self.tag_norm)

            real.append(data[i].noise_std.real/self.noise_norm)
            imag.append(data[i].noise_std.imag/self.noise_norm)

            real_list.append(real)
            imag_list.append(imag)

        return torch.FloatTensor(np.array([real_list, imag_list]))

    def parse_label(self, label):
        y_data = []

        for i in range(6):
            label_element = complex(label[i])
            y_data.append(label_element.real/self.tag_norm)
            y_data.append(label_element.imag/self.tag_norm)

        return torch.FloatTensor(np.array(y_data))


global_data_handler = DataHandler()
global_key_list = list(global_data_handler.getKey())


class DataProcessor:
    def __init__(self, multiply=1, key_list=global_key_list, normalize=None):
        self.data_handler = global_data_handler
        self.data_label_list = []
        self.data_key_list = key_list

        if normalize is None:
            mean_tag = 0.0
            mean_std = 0.0
            data_len = 0
            for data, label, key in self.data_handler:
                if key not in self.data_key_list:
                    continue

                data_len += len(data)
                for d in data:
                    mean_tag += abs(d.tag_sig)
                    mean_std += abs(d.noise_std)

            mean_tag /= data_len
            mean_std /= data_len

            self.normalize = (mean_tag, mean_std)
        else:
            self.normalize = normalize

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

                prepared_data_list.append(dataParser(data_to_parse, label, key, self.normalize))

                last_idx = i

            # handle last remaining data
            if last_idx < len(data) - SIZE_OF_DATA:
                data_to_parse = data[len(data) - SIZE_OF_DATA:len(data)]

                prepared_data_list.append(dataParser(data_to_parse, label, key, self.normalize))

        return prepared_data_list

    def __len__(self):
        return len(self.data_label_list)

    def __getitem__(self, idx):
        return self.data_label_list[idx].x_data, self.data_label_list[idx].y_data


def main():
    d = DataProcessor(multiply=5)

    # for x, y in d:
        #print(y.shape)

    print(len(d))


if __name__ == "__main__":
    main()
