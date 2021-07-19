from multiprocessing import Process, Manager

def do_work(func, para_list, num_of_thread=8):
    if len(para_list) < num_of_thread:
        num_of_thread = len(para_list)

    m = Manager()
    result_list = m.list()
    proc_list = []

    para_len = len(para_list)
    step_len = int(para_len/num_of_thread)
    remain = para_len - step_len*num_of_thread

    start_idx = 0
    end_idx = step_len

    if remain > 0:
        end_idx += 1

    def worker(para_range, result_list):
        worker_result_list = []
        for p in para_range:
            worker_result_list.append(func(*para_list[p]))
        result_list += worker_result_list

    for i in range(num_of_thread):
        proc_list.append(Process(target=worker, args=(range(start_idx, end_idx), result_list), daemon=True))

        start_idx = end_idx
        end_idx += step_len

        if (i+1) < remain:
            end_idx += 1

    for p in proc_list:
        p.start()

    for p in proc_list:
        p.join()

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