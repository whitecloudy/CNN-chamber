import cmath
import copy
import numpy as np
from DataHandler import dataParser

aug_para = (4, 4, 4)

def data_augmentation(data, label, key):
    result_list = [(data, label, key)]

    aug_result = []
    
    for d, l, k in result_list:
        aug_result += data_aug1(d, l, k)
    result_list = aug_result

    aug_result = []

    for d, l, k in result_list:
        aug_result += data_aug2(d, l, k)
    result_list = aug_result

    aug_result = []

    for d, l, k in result_list:
        aug_result += data_aug3(d, l, k)
    result_list = aug_result
    
    return result_list


def data_aug1(data, label, key):
    result_list = [(data, label, key + (0,))]

    tag_sig_divide = aug_para[0]
    
    for i in range(1, tag_sig_divide):
        fix_shift = cmath.rect(1, 2*cmath.pi*(i/tag_sig_divide)) 
        random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()*tag_sig_divide))

        shift_val = fix_shift * random_shift

        shift_key = key + (i,)
        shift_label = label * shift_val
        shift_data_list = []#copy.deepcopy(data)

        for d in data:
            shift_tag_sig = d.tag_sig * shift_val
            shift_noise_std = d.noise_std * shift_val
            shift_noise_std = complex(abs(shift_noise_std.real), abs(shift_noise_std.imag))
            shift_data_list.append(dataParser(d.phase_vec, shift_key, d.round_num, shift_noise_std, shift_tag_sig))

        result_list.append((shift_data_list, shift_label, shift_key))

    return result_list


def data_aug2(data, label, key):
    result_list = [(data, label, key + (0,))]

    phase_vec_divide = aug_para[1]
    
    for r in range(1, phase_vec_divide):
        fix_shift = cmath.rect(1, 2*cmath.pi*(r/phase_vec_divide)) 
        random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()/phase_vec_divide))

        shift_val = fix_shift * random_shift

        shift_label = label * shift_val.conjugate()
        shift_key = key + (r,)
        shift_data_list = []#copy.deepcopy(data)

        for d in data:
            shift_phase_vec = d.phase_vec * shift_val
            shift_data_list.append(dataParser(shift_phase_vec, shift_key, d.round_num, d.noise_std, d.tag_sig))

        result_list.append((shift_data_list, shift_label, shift_key))

    return result_list

def data_aug3(data, label, key):
    result_list = [(data, label, key + (0,))]

    shuffle_candidate = [[2, 1, 0, 5, 4, 3], [3, 4, 5, 0, 1, 2], [5, 4, 3, 2, 1, 0]]

    for shuffle in shuffle_candidate:
        shuffle_label = np.array([label[shuffle[i]] for i in range(6)])

        shuffle_key = key + (shuffle,)
        shuffle_data_list = []

        for d in data:
            shuffle_phase_vec = [d.phase_vec[shuffle[i]] for i in range(6)]
            shuffle_data_list.append(dataParser(shuffle_phase_vec, shuffle_key, d.round_num, d.noise_std, d.tag_sig))

        result_list.append((shuffle_data_list, shuffle_label, shuffle_key))

    return result_list

