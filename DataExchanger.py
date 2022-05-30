import torch
import cmath
import numpy as np

class data_parser:
    def __init__(self, data):
        def parsing(data):
            x_data = []
            for d in data:
                x_row = np.append(d[0], d[1])
                x_row = np.append(x_row, d[2])
                x_data.append(x_row)

            x_data = np.array(x_data)

            return torch.FloatTensor([x_data.real, x_data.imag])

        def calculate_heur(data):
            phase_vec = []
            tag_sig = []
            for d in data:
                phase_vec.append(d[0])
                tag_sig.append([d[1]])
            
            W = np.matrix(phase_vec)
            A = np.matrix(tag_sig)
            W_inv = (W.H*W).I * W.H

            result = np.array(W_inv * A).reshape((1, 6))

            return torch.FloatTensor([result.real, result.imag]).reshape(12,)

        def select_best(data):
            max_phase_vec = None
            max_tag_sig = float("-inf")
            for d in data:
                phase_vec = d[0]
                tag_sig = d[1]

                if abs(tag_sig) > max_tag_sig:
                    max_tag_sig = abs(tag_sig)
                    max_phase_vec = phase_vec

            result = np.array(max_phase_vec).reshape((1, 6)).conj()
            return torch.FloatTensor([result.real, result.imag]).reshape(12,)

        self.x_data = parsing(data)
        self.heur_data = calculate_heur(data)
        self.select_best_data = select_best(data)



import socket
class DataExchanger:
    def __init__(self, IP='', port=11045):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.sock.bind((IP, port))
        self.recv_addr = None


    def send_channel(self, channel):
        print("Send channel")
        print(channel)
        for c in channel:
            print(int(cmath.phase(c)/cmath.pi * 180))
        send_str = ""

        for i, c in enumerate(channel):
            send_str += (str(c.real) + ',')
            send_str += (str(c.imag))

            if (len(channel)-1) != i:
                send_str += ','

        send_byte = send_str.encode('utf-8')
        self.sock.sendto(send_byte, self.recv_addr)


    def recv_data(self, wait_data_len):
        print(wait_data_len)
        x_matrix = []
        recv_count = 0
        while recv_count < wait_data_len:
            recv_byte, addr = self.sock.recvfrom(4096)
            self.recv_addr = addr

            recv_str = recv_byte[:-1].decode('utf-8')

            if recv_str[0] == '1':
                print("End")
                return None, None
            elif recv_str[0] == '2':
                print("Reset")
                x_matrix = []
                recv_count = 0
                continue
            
            recv_str = recv_str[1:]

            real = 0.0
            imag = 0.0
            x_row = []

            for j, f in enumerate(recv_str.split(',')):
                if j % 2 == 0:
                    real = float(f)
                else:
                    imag = float(f)
                    x_row.append(complex(real, imag))
            # phase_vec, tag_sig, noise_std
            x_data = (x_row[0:6], x_row[6], x_row[7])

            x_matrix.append(x_data)
            recv_count += 1

        parsed_data = data_parser(x_matrix)

        return parsed_data.x_data, parsed_data.heur_data, parsed_data.select_best_data
