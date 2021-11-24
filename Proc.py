from multiprocessing import Process, Manager
import sys
import numpy as np

class pseudo_list:
    def __init__(self, l_data=None):
        self.l_data = []
        self.length = 0

        if l_data is not None:
            self.append(l_data)

    def append(self, data):
        if type(data) is list:
            self.l_data.append((len(data), data))
            self.length += len(data)
        elif type(data) is pseudo_list:
            self.l_data += data.l_data
            self.length += data.length
        else:
            sys.exit(1)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        for d_len, data in self.l_data:
            if d_len > idx:
                return data[idx]
            else:
                idx -= d_len

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx >= self.length:
            raise StopIteration

        return_data = self.__getitem__(self.cur_idx)
        self.cur_idx += 1

        return return_data


def do_work(func, para_list, num_of_thread=8):
    if len(para_list) < num_of_thread:
        num_of_thread = len(para_list)

    m = Manager()
    result_queue = m.Queue()
    proc_list = []

    para_len = len(para_list)
    step_len = int(para_len/num_of_thread)
    remain = para_len - step_len*num_of_thread

    start_idx = 0
    end_idx = step_len

    if remain > 0:
        end_idx += 1

    def worker(i, para_range, result_queue):
        worker_result_list = []
        for p in para_range:
            worker_result_list += func(*para_list[p])
        result_queue.put((i, worker_result_list))

    result_list = []

    for i in range(num_of_thread):
        proc_list.append(Process(target=worker, args=(i, range(start_idx, end_idx), result_queue), daemon=True))

        start_idx = end_idx
        end_idx += step_len

        if (i+1) < remain:
            end_idx += 1

    for p in proc_list:
        p.start()

    result_list = pseudo_list()
    for i in range(num_of_thread):
        result = result_queue.get()
        result_list.append(result[1])
        proc_list[result[0]].join()

    for p in proc_list:
        p.join()

    print(len(result_list))

    return result_list


if __name__ == "__main__":
    def test_work(num, data):
        # print(num)
        data += 10
        return data

    datas = list(range(9999))
    for i, d in enumerate(datas):
        datas[i] = (i, d)
        
    result = do_work(test_work, datas)
    print(result)
    print(len(result))
