import torch
import torch.nn.functional as F

def make_complex(x):
    input2 = torch.tensor_split(x, 2, dim=1)
    complex_x = torch.complex(input2[0], input2[1])
    return complex_x


def complex_cosine_sim_loss(x1, x2):
    x1_c = make_complex(x1)
    x2_c = make_complex(x2)

    #print(x1_c[0])
    #print(x2_c[0])
    dot_pro = x1_c * x2_c.conj()

    #print(dot_pro[0])

    x1_abs = abs(x1_c)
    x2_abs = abs(x2_c)

    cos_sim = dot_pro/(x1_abs * x2_abs)

    return 1 - abs(torch.mean(cos_sim))

if __name__=="__main__":
    x1 = torch.randn(6,12)
    x2 = torch.randn(6,12)

    print(complex_cosine_sim_loss(x1, x2))
    print(complex_cosine_sim_loss(x1, 5*x2))
