import glob
import csv
import cmath
import numpy as np
import copy

DATADIR = "data/*"

class dataParser:
    """
    Parsing Data
    """
    def __init__(self, phase_vec, data_kind, round_num, noise_std, tag_sig):
        self.phase_vec = np.copy(phase_vec)
        self.data_kind = data_kind
        self.round_num = round_num
        self.noise_std = noise_std
        self.tag_sig = tag_sig
        self.x_row = np.append(np.append(self.phase_vec, self.tag_sig), self.noise_std)

    @classmethod
    def from_row_filename(cls, row, filename):
        phase_vec = []
        for i in range(6):
            phase_vec.append(cmath.rect(1, int(row[i])/180 * cmath.pi))
        phase_vec = np.array(phase_vec)

        data_kind = (filename, row[6], row[7], row[8], row[9])
        round_num = int(row[10])
        noise_std = complex(float(row[11]), float(row[12]))
        tag_sig = complex(-float(row[13]), -float(row[14]))

        return cls(phase_vec, data_kind, round_num, noise_std, tag_sig)

    @classmethod
    def copySrc(cls, src):
        return cls(src.phase_vec, src.data_kind, src.round_num, src.noise_std, src.tag_sig)


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
                        member = dataParser.from_row_filename(row, filename)

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

        self.key_list = list(sorted(self.data_dict.keys()))

    def getKey(self):
        return self.data_dict.keys()

    def getCha(self, key):
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

        channel = (W.getH() * W).getI() * (W.getH() * A)

        return channel


    def getLabel(self, key):
        channel = self.getCha(key)
        
        """
        for i, cha in enumerate(channel):
            channel[i] = (cha/abs(cha))
        """
        
        return np.array(channel)


    def getData(self, key):
        return self.data_dict[key]

    def evalLabel(self, key, printFlag=False):
        H = self.getCha(key)
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
        key = self.key_list[idx]
        return self.getData(key), self.getLabel(key), key


def main():
    data = DataHandler()

    for d, l, k in data:
        for amp in abs(l):
            print(float(amp), ",", len(d))

if __name__ == "__main__":
    main()
