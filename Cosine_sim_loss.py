import torch
import torch.nn.functional as F

def make_complex(x : torch.Tensor):
    input2 = torch.tensor_split(x, 2, dim=(x.dim()-1))
    complex_x = torch.complex(input2[0], input2[1])
    return complex_x


def spread_complex(x):
    result_tensor = torch.cat((x.real, x.imag), dim=(x.dim()-1))
    return result_tensor


def cosine_sim(x1_c, x2_c, dim=1):
    dot_pro = x1_c * x2_c.conj()

    x1_abs = abs(x1_c)
    x2_abs = abs(x2_c)

    cos_sim = dot_pro/(x1_abs * x2_abs)

    return abs(torch.mean(cos_sim, dim=dim))

def amp_expectation(x_c : torch.Tensor, label_c : torch.Tensor):
    x_c_norm = x_c / torch.abs(x_c)

    amp_expected = torch.abs(torch.sum((x_c_norm.conj() * label_c), dim=(x_c_norm.dim()-1)))
    
    return amp_expected


def norm_amp_expectation(x_c : torch.Tensor, label_c : torch.Tensor):
    amp_expected = amp_expectation(x_c, label_c)

    max_amp_expected = torch.sum(abs(label_c), dim=(label_c.dim()-1))

    return amp_expected/max_amp_expected


def norm_amp_loss(x : torch.Tensor, label : torch.Tensor):
    x_c = make_complex(x)
    label_c = make_complex(label)

    return torch.mean(norm_amp_expectation(x_c, label_c))
    

def MSE(x1, x2, dim=1):
    error = x1 - x2
    return torch.mean(error * error, dim=dim)


def MSE_normalize(mse_loss, label):
    complexed_label = make_complex(label)
    label_abs = torch.sum(torch.abs(complexed_label), dim=1)
    mse_loss = torch.sum(mse_loss, dim=1)
    normalized_mse = torch.sum(mse_loss / label_abs)

    return normalized_mse


def complex_cosine_sim_loss(x1, x2):
    x1_c = make_complex(x1)
    x2_c = make_complex(x2)

    cos_sim = cosine_sim(x1_c, x2_c)

    return torch.mean(1 - cos_sim)


def cos_mse_mix_loss(x1, x2):
    x1_c = make_complex(x1)
    x2_c = make_complex(x2)
    
    cos_sim = 1 - cosine_sim(x1_c, x2_c)
    mse = MSE(x1, x2)
    
    return torch.mean(cos_sim * mse)


if __name__=="__main__":
    x1 = torch.randn(1000, 12)
    x2 = torch.randn(1000, 12)

    print(complex_cosine_sim_loss(x1, x2))
