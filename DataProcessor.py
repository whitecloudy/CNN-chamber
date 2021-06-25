from DataHandler import DataHandler
import numpy as np
import torch
import random
import time
import cmath
import copy

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
            norm = float(self.channel_norm[i])
            y_data.append(label_element.real/norm)
            y_data.append(label_element.imag/norm)

        return torch.FloatTensor(np.array(y_data))


global_data_handler = DataHandler()
global_key_list = list(global_data_handler.getKey())


class DataProcessor:
    def __init__(self, multiply=1, key_list=global_key_list, normalize=None):
        self.data_handler = global_data_handler
        self.output_list = []
        self.key_list = key_list
        self.data_label_key_list = []
        
        # data augmentation
        for data, label, key in self.data_handler:
            if key not in self.key_list:
                continue
            self.data_label_key_list.append((data, label, key))
            self.data_label_key_list += self.data_augmentation(data=data,
                                                               label=label,
                                                               key=key)

        print("Data Augmentation Complete")

        # config normalize value
        if normalize is None:
            self.normalize = self.calculate_normalize()
        else:
            self.normalize = normalize
        
        # prepare output data
        self.prepare_data(multiply)


    def prepare_data(self, multiply):
        self.output_list = []
        for data, label, key in self.data_label_key_list:
            self.output_list += self.make_output_data(data, label, key, multiply)


    def calculate_normalize(self):
        mean_tag = 0.0
        mean_std = 0.0
        mean_label = np.zeros((6, 1))
        avg_tag = 0.0+0.0j
        avg_std = 0.0+0.0j
        avg_label = np.zeros((6, 1), dtype='complex128')

        data_len = 0
        for data, label, key in self.data_label_key_list:
            if key not in self.key_list:
                continue

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

        mean_label /= len(self.data_handler)
        avg_label /= len(self.data_handler)

        return (mean_tag, mean_std, mean_label, avg_tag, avg_std, avg_label)


    def make_output_data(self, data, label, key, multiply=1):
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

    def data_augmentation(self, data, label, key):
        original_data = copy.deepcopy(data)
        original_label = copy.deepcopy(label)
        original_key = copy.deepcopy(key)

        result_list = []

        tag_sig_divide = 4
        phase_vec_divide = 4
        
        # add original data
        for i in range(tag_sig_divide):
            data = copy.deepcopy(original_data)
            label = copy.deepcopy(original_label)
            key = copy.deepcopy(original_key)

            shift_val = cmath.rect(1, cmath.pi*(i/(tag_sig_divide/2))) * cmath.rect(1, cmath.pi*(np.random.rand()*2/tag_sig_divide))

            for d in data:
                d.tag_sig *= shift_val
                d.noise_std *= shift_val
                d.noise_std = complex(abs(d.noise_std.real), abs(d.noise_std.imag))

            label *= shift_val

            shift_key = key + (i, 0)
    
            result_list.append((data, label, shift_key))
            
            for r in range(1, phase_vec_divide):
                random_shift_val = cmath.rect(1, cmath.pi*(r/(phase_vec_divide/2))) * cmath.rect(1, cmath.pi*(np.random.rand()*2/phase_vec_divide))
                for d in data:
                    d.phase_vec *= random_shift_val
                label *= random_shift_val.conjugate()

                random_key = key + (i, r)
                result_list.append((data, label, random_key))

        return result_list

    def __len__(self):
        return len(self.output_list)

    def __getitem__(self, idx):
        return self.output_list[idx].x_data, self.output_list[idx].y_data


def main():
    d = DataProcessor(multiply=5)

    # for x, y in d:
        #print(y.shape)

    print(len(d))


if __name__ == "__main__":
    main()
