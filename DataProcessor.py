import numpy as np
import torch
import random
import time
import cmath
import copy

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
                label_element = H[i]
                norm = float(self.channel_norm[i])
                result_data.append(label_element.real/norm)
                result_data.append(label_element.imag/norm)

            return torch.FloatTensor(np.array(result_data))
        except np.linalg.LinAlgError:
            return self.y_data

       
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
        return_val = torch.FloatTensor(np.array([real_list, imag_list]))

        return return_val


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

            if key[3] == ' directionalrefine':
                continue

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
        random.seed(time.time())
        random.shuffle(self.data_label_key_list)

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
        result_list = []
        result_list.append((data, label, key))

        aug_result = []

        for d, l, k in result_list:
            aug_result += self.data_aug1(d, l, k)
        result_list += aug_result

        aug_result = []

        for d, l, k in result_list:
            aug_result += self.data_aug2(d, l, k)
        result_list += aug_result

        return result_list


    def data_aug1(self, data, label, key):
        result_list = []

        tag_sig_divide = 8
        
        for i in range(1, tag_sig_divide):
            fix_shift = cmath.rect(1, 2*cmath.pi*(i/tag_sig_divide)) 
            random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()*tag_sig_divide))

            shift_val = fix_shift * random_shift
            shift_data_list = []

            for d in data:
                shift_data = copy.deepcopy(d)
                shift_data.tag_sig *= shift_val
                shift_data.noise_std *= shift_val
                shift_data.noise_std = complex(abs(d.noise_std.real), abs(d.noise_std.imag))
                shift_data_list.append(shift_data)

            shift_label = label * shift_val

            shift_key = key + (i,)
    
            result_list.append((shift_data_list, shift_label, shift_key))

        return result_list


    def data_aug2(self, data, label, key):
        result_list = []

        phase_vec_divide = 8
        
        for r in range(1, phase_vec_divide):
            fix_shift = cmath.rect(1, 2*cmath.pi*(r/phase_vec_divide)) 
            random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()/phase_vec_divide))

            shift_val = fix_shift * random_shift

            shift_data_list = []

            for d in data:
                shift_data = copy.deepcopy(d)
                shift_data.phase_vec *= shift_val
                shift_data_list.append(shift_data)

            shift_label = label * shift_val.conjugate()

            shift_key = key + (r,)

            result_list.append((shift_data_list, shift_label, shift_key))

        return result_list

    def data_aug3(self, data, label, key):
        result_list = []

        shuffle_candidate = list(permutations(range(6), 6))
        shuffle_candidate.remove((0, 1, 2, 3, 4, 5))

        for shuffle in shuffle_candidate:
            pass

        return result_list


    def __len__(self):
        return len(self.output_list)

    def __getitem__(self, idx):
        return self.output_list[idx].x_data, self.output_list[idx].y_data, self.output_list[idx].heur_data


def main():
    d = DataProcessor(multiply=1)

    # for x, y in d:
        #print(y.shape)

    print(len(d))
    print(d.normalize)


if __name__ == "__main__":
    main()
