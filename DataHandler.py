import glob
import csv
import cmath
import numpy as np

DATADIR = "data/*"

class dataParser:
    """
    Parsing Data
    """
    def __init__(self, row, filename):
        self.phase_vec = []

        for i in range(6):
            self.phase_vec.append(cmath.rect(1, int(row[i])/180 * cmath.pi))
        self.phase_vec = np.array(self.phase_vec)

        self.data_kind = (filename, row[6], row[7], row[8], row[9])
        self.round_num = int(row[10])
        self.noise_std = complex(float(row[11]), float(row[12]))
        self.tag_sig = complex(-float(row[13]), -float(row[14]))


class DataHandler:
    """
    Beamforming Data handling Class
    :len: length of data lists
    :getitem : return list of datas
    """
    def __init__(self):
        filelist = glob.glob(DATADIR)
        self.data_dict = {}
        self.opt_data_dict = {}

        for filename in filelist:
            try:
                with open(filename, "r") as log_file:
                    log_reader = csv.reader(log_file)
                    first = True
                    tmp_dict = {}
                    for row in log_reader:
                        if first:
                            first = False
                            continue
                        member = dataParser(row, filename)

                        if member.data_kind in tmp_dict:
                            tmp_dict[member.data_kind].append(member)
                        else:
                            tmp_dict[member.data_kind] = [member]
                for kind in tmp_dict:
                    tmp_dict[kind] = sorted(tmp_dict[kind], key=lambda x: x.round_num)
                    if filename.find("opt") != -1:
                        self.opt_data_dict[kind] = tmp_dict[kind]
                    else:
                        self.data_dict[kind] = tmp_dict[kind]
            except IsADirectoryError:
                pass
    def getKey(self):
        return self.data_dict.keys()

    def getLabel(self, key):
        W = []
        A = []

        for data in self.data_dict[key]:
            W.append(data.phase_vec)
            A.append([data.tag_sig])
        
        W = np.matrix(W)
        A = np.matrix(A)

        """
        u, s, vh = np.linalg.svd(W)
        get_val = 0.0
        for s_elem in s:
            get_val += (1/s_elem)**2
        print(get_val**0.5)
        #s_inv = s.getI()
        """

        return (W.getH() * W).getI() * (W.getH() * A)

    def getData(self, key):
        return self.data_dict[key]

    def evalLabel(self, key, printFlag=False):
        H = self.getLabel(key)
        eval_value = []
        sig_data = []
        exp_data = []


        for data in self.data_dict[key]:
            real_sig = data.tag_sig
            expect_sig = data.phase_vec * H
            sig_data.append(abs(real_sig))
            exp_data.append(abs(expect_sig))
            eval_value.append(abs(complex(real_sig) - complex(expect_sig))/abs(expect_sig))
            #eval_value.append(abs(cmath.phase(real_sig/expect_sig)/cmath.pi * 180))

        sig_data = np.array(sig_data)
        eval_value = np.array(eval_value)
        if printFlag:
            print(np.mean(eval_value))
            #print(np.mean(sig_data))
            #print(np.std(eval_value))
            #print(np.std(sig_data)/np.mean(sig_data))

        return np.mean(eval_value), len(self.data_dict[key])

    def __len__(self):
        return len(self.data_dict)

    def __contains__(self, key):
        return key in self.data_dict

    def __getitem__(self, idx):
        key = list(self.data_dict.keys())[idx]
        return self.getData(key), self.getLabel(key), key


def main():
    d = DataHandler()
    i = 0
    
    for i in d.data_dict.keys():
        print(i, len(d.data_dict[i]))
        d.evalLabel(i, True)
        print()
    # for data, label in d:

        

if __name__ == "__main__":
    main()
