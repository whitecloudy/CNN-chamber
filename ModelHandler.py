import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def model_selector(model_name, row_size):
    if model_name == 'Net':
        return Net(row_size)
    elif model_name == 'conv1d':
        return Net_with1d(row_size)
    elif model_name == 'conv1d_without_ls':
        return Net_withoutLS(row_size)
    elif model_name == 'conv1d_without_cnn':
        return Net_withoutRow(row_size)
    elif model_name == 'transformer':
        return Net_transformer_encoder(row_size)
    elif model_name == 'transformer_encoder_only':
        return Net_transformer_encoder_only(row_size)
    elif model_name == 'transformer_ls_stack':
        return Net_transformer_encoder_LSstack(row_size)
    else:
        return -1


class Net(nn.Module):
    def __init__(self, row_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1) #input is 9 * 8 * 2
        row_size -= 2
        self.conv2 = nn.Conv2d(64, 64, 3, 1) # input is 7 * 6 * 64
        row_size -= 2
        self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(row_size * 4 * 64 + 256, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.heur_fc1(x1)
        x1 = self.heur_batch(x1)
        x1 = F.leaky_relu(x1)
        x = torch.cat((x, x1), 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x


class Net_with1d(nn.Module):
    def __init__(self, row_size):
        super(Net_with1d, self).__init__()
        self.first_fc = nn.Linear(16, 8 * 8)

        self.conv1 = nn.Conv2d(8, 64, 3, 1) #input is 9 * 8 * 2
        row_size -= 2
        self.conv2 = nn.Conv2d(64, 64, 3, 1) # input is 7 * 6 * 64
        row_size -= 2
        self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(row_size * 4 * 64 + 256, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.first_fc.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = torch.tensor_split(x, 2, dim=1)
        x = torch.cat((x[0], x[1]), dim=3)
        x = self.first_fc(x)
        x = torch.cat(torch.tensor_split(x, 8, dim=3), dim=1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.heur_fc1(x1)
        x1 = self.heur_batch(x1)
        x1 = F.leaky_relu(x1)
        x = torch.cat((x, x1), 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x


class Net_withoutRow(nn.Module):
    def __init__(self, row_size):
        super(Net_withoutRow, self).__init__()
        self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = self.heur_fc1(x1)
        x = self.heur_batch(x)
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x

class Net_withoutLS(nn.Module):
    def __init__(self, row_size):
        super(Net_withoutLS, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, 1) #input is 9 * 8 * 2
        row_size -= 2
        self.conv2 = nn.Conv2d(64, 64, 3, 1) # input is 7 * 6 * 64
        row_size -= 2
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x


class Net_transformer_encoder(nn.Module):
    def __init__(self, row_size):
        super(Net_transformer_encoder, self).__init__()
        self.first_fc = nn.Linear(16, 8 * 8)
        self.d_model = 64
        encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=256, dropout=0.1, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=3)
        self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(row_size * self.d_model + 256, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.first_fc.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = torch.tensor_split(x, 2, dim=1)
        x = torch.cat((x[0], x[1]), dim=3)
        x = self.first_fc(x)
        x = torch.squeeze(x, dim=1)
        x = x.permute(1, 0, 2)
        #x = x * math.sqrt(self.d_model)
        x = self.transformer_encoder(x)
        x = F.gelu(x)
        x = x.permute(1, 0, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x1 = self.heur_fc1(x1)
        x1 = self.heur_batch(x1)
        x1 = F.leaky_relu(x1)
        x = torch.cat((x, x1), 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x

class Net_transformer_encoder_only(nn.Module):
    def __init__(self, row_size):
        super(Net_transformer_encoder_only, self).__init__()
        self.first_fc = nn.Linear(16, 8 * 8)
        self.d_model = 64
        encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=256, dropout=0.1, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=3)
        #self.heur_fc1 = nn.Linear(12, 256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.batch3 = nn.BatchNorm1d(1024)
        #self.heur_batch = nn.BatchNorm1d(256)

        #self.fc1 = nn.Linear(row_size * self.d_model + 256, 1024) 
        self.fc1 = nn.Linear(row_size * self.d_model, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.first_fc.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        #torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = torch.tensor_split(x, 2, dim=1)
        x = torch.cat((x[0], x[1]), dim=3)
        x = self.first_fc(x)
        x = torch.squeeze(x, dim=1)
        x = x.permute(1, 0, 2)
        #x = x * math.sqrt(self.d_model)
        x = self.transformer_encoder(x)
        x = F.gelu(x)
        x = x.permute(1, 0, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #x1 = self.heur_fc1(x1)
        #x1 = self.heur_batch(x1)
        #x1 = F.leaky_relu(x1)
        #x = torch.cat((x, x1), 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x


class Net_transformer_encoder_LSstack(nn.Module):
    def __init__(self, row_size):
        super(Net_transformer_encoder_LSstack, self).__init__()
        self.first_fc = nn.Linear(16, 8 * 8)
        self.d_model = 64
        encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=256, dropout=0.1, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=3)
        self.heur_fc1 = nn.Linear(12, 64)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.batch3 = nn.BatchNorm1d(1024)
        self.heur_batch = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear((row_size+1) * self.d_model, 1024) # 21 * 2 * 64
        #self.fc1 = nn.Linear(row_size * 4 * 64, 1024) # 21 * 2 * 64
        self.fc2 = nn.Linear(1024, 12)

        torch.nn.init.xavier_uniform_(self.first_fc.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.heur_fc1.weight)


    def forward(self, x, x1):
        #x = torch.tensor_split(x, (7, ), dim=3)
        #x = x[0]
        x = torch.tensor_split(x, 2, dim=1)
        x = torch.cat((x[0], x[1]), dim=3)
        x = self.first_fc(x)
        x1 = self.heur_fc1(x1)
        x1 = torch.unsqueeze(x1, dim=1)
        x = torch.squeeze(x, dim=1)
        x = torch.cat((x, x1), dim=1)
        x = x.permute(1, 0, 2)
        #x = x * math.sqrt(self.d_model)
        x = self.transformer_encoder(x)
        x = F.gelu(x)
        x = x.permute(1, 0, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        #output = F.log_softmax(x, dim=1)

        return x
