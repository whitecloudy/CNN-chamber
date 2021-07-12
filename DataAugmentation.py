import cmath
import copy
import numpy as np

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

    tag_sig_divide = 4
    
    for i in range(1, tag_sig_divide):
        fix_shift = cmath.rect(1, 2*cmath.pi*(i/tag_sig_divide)) 
        random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()*tag_sig_divide))

        shift_val = fix_shift * random_shift
        shift_data_list = copy.deepcopy(data)

        for shift_data in shift_data_list:
            shift_data.tag_sig *= shift_val
            shift_data.noise_std *= shift_val
            shift_data.noise_std = complex(abs(shift_data.noise_std.real), abs(shift_data.noise_std.imag))

        shift_label = label * shift_val

        shift_key = key + (i,)

        result_list.append((shift_data_list, shift_label, shift_key))

    return result_list


def data_aug2(data, label, key):
    result_list = [(data, label, key + (0,))]

    phase_vec_divide = 4
    
    for r in range(1, phase_vec_divide):
        fix_shift = cmath.rect(1, 2*cmath.pi*(r/phase_vec_divide)) 
        random_shift = cmath.rect(1, 2*cmath.pi*(np.random.rand()/phase_vec_divide))

        shift_val = fix_shift * random_shift

        shift_data_list = copy.deepcopy(data)

        for shift_data in shift_data_list:
            shift_data.phase_vec *= shift_val

        shift_label = label * shift_val.conjugate()

        shift_key = key + (r,)

        result_list.append((shift_data_list, shift_label, shift_key))

    return result_list

def data_aug3(data, label, key):
    result_list = [(data, label, key + (0,))]

    shuffle_candidate = [[2, 1, 0, 5, 4, 3], [3, 4, 5, 0, 1, 2], [5, 4, 3, 2, 1, 0]]

    for shuffle in shuffle_candidate:
        shuffle_data_list = copy.deepcopy(data)

        for ori_idx, shuffle_data in enumerate(shuffle_data_list):
            d = data[ori_idx]
            i = 0
            for idx in shuffle:
                shuffle_data.phase_vec[i] = d.phase_vec[idx]
                i += 1

        shuffle_label = copy.deepcopy(label)

        i = 0
        for idx in shuffle:
            shuffle_label[i] = label[idx]

        shuffle_key = key + (shuffle,)

        result_list.append((shuffle_data_list, shuffle_label, shuffle_key))

    return result_list

