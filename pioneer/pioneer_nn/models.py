import torch
import torch.nn as nn
import torch.nn.functional as F
from .kafnets import KAF
from .arma_conv_KAF import *


class GCN_RNN_NET(nn.Module):
    def __init__(self, gcn_input_dim, gcn_output_dim, gcn_layers, gcn_drop_prob, rnn_input_dim, rnn_hidden_dim, rnn_layers, rnn_drop_prob, rnn_batch_size, drop_prob, output_size, act, boundary, D):
        super(GCN_RNN_NET, self).__init__()
        self.gcn_input_dim = gcn_input_dim
        self.gcn_output_dim = gcn_output_dim
        self.gcn_layers = gcn_layers
        self.gcn_drop_prob = gcn_drop_prob
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.rnn_drop_prob = rnn_drop_prob
        self.rnn_batch_size = rnn_batch_size
        self.drop_prob = drop_prob
        self.output_size = output_size
        self.act = act
        self.boundary = boundary
        self.D = D

        # RNN
        self.rnn_unit = nn.GRU(self.rnn_input_dim, self.rnn_hidden_dim, self.rnn_layers, batch_first = True, dropout = self.rnn_drop_prob, bidirectional = True)
        for weight in self.rnn_unit.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.xavier_uniform_(weight)

        # GCN layers
        self.gcnLayers = ARMAConvLayers(self.gcn_input_dim, self.gcn_output_dim, self.gcn_layers, self.gcn_drop_prob, D = self.D, boundary = self.boundary, kernel = self.act)

        # FC layers
        self.fc1 = nn.Linear((self.gcn_output_dim + self.rnn_hidden_dim * 2) * 3, 512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.KAF1 = KAF(512, D = self.D, boundary = self.boundary, kernel = self.act)
        self.fc2 = nn.Linear(512, 512)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.KAF2 = KAF(512, D = self.D, boundary = self.boundary, kernel = self.act)
        self.fc3 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.KAF3 = KAF(256, D = self.D, boundary = self.boundary, kernel = self.act)
        self.fc4 = nn.Linear(256, output_size)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, gcn_feat, edge_index, partner_gcn_feat, partner_edge_index, res_samples, rnn_feat, partner_rnn_feat, h, res_id, partner_res_id):
        gcn_out = self.gcnLayers(gcn_feat, edge_index)
        partner_gcn_out = self.gcnLayers(partner_gcn_feat, partner_edge_index)
        
        rnn_feat = rnn_feat.float()
        rnn_out, hidden = self.rnn_unit(rnn_feat, h)
        rnn_out = rnn_out.squeeze()
        rnn_out = rnn_out[res_id]

        partner_rnn_feat = partner_rnn_feat.float()
        partner_rnn_out, partner_hidden = self.rnn_unit(partner_rnn_feat, h)
        partner_rnn_out = partner_rnn_out.squeeze()
        partner_rnn_out = partner_rnn_out[partner_res_id]
        emb = torch.cat((gcn_out, rnn_out), 1)
        partner_emb = torch.cat((partner_gcn_out, partner_rnn_out), 1)

        samples = emb[res_samples]
        num_samples = samples.size(0)

        info = torch.mean(emb, 0)
        partner_info = torch.mean(partner_emb, 0)

        info = (info.expand(num_samples, self.gcn_output_dim + self.rnn_hidden_dim * 2)).float()
        partner_info = (partner_info.expand(num_samples, self.gcn_output_dim + self.rnn_hidden_dim * 2)).float()

        embeddings = torch.cat((samples, info, partner_info), 1)

        dense_output = self.dropout(self.KAF1(self.fc1(embeddings)))
        dense_output = self.dropout(self.KAF2(self.fc2(dense_output)))
        dense_output = self.dropout(self.KAF3(self.fc3(dense_output)))
        prediction = self.fc4(dense_output)
        prediction = self.fc4(dense_output)

        return prediction

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_layers * 2, self.rnn_batch_size, self.rnn_hidden_dim).zero_()
        return hidden

class RNN_NET(nn.Module):
    def __init__(self, unit, input_dim, output_size, hidden_dim, n_layers, drop_prob):
        super(RNN_NET, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        if unit == 'LSTM':
            self.rnn_unit = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = drop_prob, bidirectional = True)
        if unit == 'GRU':
            self.rnn_unit = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = drop_prob, bidirectional = True)
        for weight in self.rnn_unit.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.xavier_uniform_(weight)

        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(self.hidden_dim * 6, 512)
        self.KAF1 = KAF(512, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 256)
        self.KAF2 = KAF(256, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 128)
        self.KAF3 = KAF(128, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(128, self.output_size)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, unit, p1, p2, res_samples, hidden):
        p1 = p1.float()
        p1_out, p1_hidden = self.rnn_unit(p1, hidden)
        p1_out = p1_out.squeeze()
        p1_info = torch.mean(p1_out, 0)

        if unit == 'LSTM':
            p1_hidden = p1_hidden[0]

        p2 = p2.float()
        p2_out, p2_hidden = self.rnn_unit(p2, hidden)
        p2_out = p2_out.squeeze()
        p2_info = torch.mean(p2_out, 0)

        if unit == 'LSTM':
            p2_hidden = p2_hidden[0]

        p1_samples = p1_out[res_samples]
        num_samples = p1_samples.size(0)

        p1_info_ext = (p1_info.expand(num_samples, self.hidden_dim * 2)).float()
        p2_info_ext = (p2_info.expand(num_samples, self.hidden_dim * 2)).float()

        p1_embeddings = torch.cat((p1_samples, p1_info_ext, p2_info_ext), 1)

        dense_output = self.KAF1(self.fc1(p1_embeddings))
        dense_output = self.KAF2(self.fc2(dense_output))
        dense_output = self.KAF3(self.fc3(dense_output))

        prediction = self.fc4(dense_output)

        return prediction

    def init_hidden(self, unit, batch_size):
        weight = next(self.parameters()).data
        if unit == 'LSTM':
            hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        if unit == 'GRU':
            hidden = weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_()
        return hidden

class STR_SEQ_NET1(nn.Module):
    def __init__(self, gcn_input_dim, rnn_input_dim):
        super(STR_SEQ_NET1, self).__init__()
        self.gcn_input_dim = gcn_input_dim
        self.gcn_output_dim = 128
        self.gcn_layers = 2
        self.gcn_drop_prob = 0
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = 128
        self.rnn_layers = 2
        self.rnn_drop_prob = 0.5
        self.rnn_batch_size = 1
        self.drop_prob = 0.48600144
        self.output_size = 2

        # RNN
        self.rnn_unit = nn.GRU(self.rnn_input_dim, self.rnn_hidden_dim, self.rnn_layers, batch_first = True, dropout = self.rnn_drop_prob, bidirectional = True)
        for weight in self.rnn_unit.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.xavier_uniform_(weight)

        # GCN layers
        self.gcnLayers = ARMAConvLayers(self.gcn_input_dim, self.gcn_output_dim, self.gcn_layers, self.gcn_drop_prob, D = 20, boundary = 150, kernel = 'gaussian')

        # FC layers
        self.fc1 = nn.Linear((self.gcn_output_dim + self.rnn_hidden_dim * 2) * 3, 512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.KAF1 = KAF(512, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc2 = nn.Linear(512, 512)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.KAF2 = KAF(512, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc3 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.KAF3 = KAF(256, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc4 = nn.Linear(256, self.output_size)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, gcn_feat, edge_index, rnn_feat, h, res_id):
        gcn_out = self.gcnLayers(gcn_feat, edge_index)
        rnn_feat = rnn_feat.float()
        rnn_out, hidden = self.rnn_unit(rnn_feat, h)
        rnn_out = rnn_out.squeeze()
        rnn_out = rnn_out[res_id]
        emb = torch.cat((gcn_out, rnn_out), 1)
        return emb

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_layers * 2, self.rnn_batch_size, self.rnn_hidden_dim).zero_()
        return hidden

class STR_SEQ_NET2(nn.Module):
    def __init__(self, unit, input_dim):
        super(STR_SEQ_NET2, self).__init__()
        self.output_size = 2
        self.n_layers = 2
        self.hidden_dim = 256
        self.drop_prob = 0.5

        if unit == 'LSTM':
            self.rnn_unit = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = self.drop_prob, bidirectional = True)
        if unit == 'GRU':
            self.rnn_unit = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = self.drop_prob, bidirectional = True)
        for weight in self.rnn_unit.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.xavier_uniform_(weight)

        self.dropout = nn.Dropout(self.drop_prob)
        self.fc1 = nn.Linear(self.hidden_dim * 6, 512)
        self.KAF1 = KAF(512, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 256)
        self.KAF2 = KAF(256, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 128)
        self.KAF3 = KAF(128, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(128, self.output_size)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, unit, p1, hidden):
        p1 = p1.float()
        p1_out, p1_hidden = self.rnn_unit(p1, hidden)
        p1_out = p1_out.squeeze()
        p1_info = torch.mean(p1_out, 0)

        if unit == 'LSTM':
            p1_hidden = p1_hidden[0]

        return p1_info

    def init_hidden(self, unit, batch_size):
        weight = next(self.parameters()).data
        if unit == 'LSTM':
            hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        if unit == 'GRU':
            hidden = weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_()
        return hidden

class STR_SEQ_NET(nn.Module):
    def __init__(self, drop_prob):
        super(STR_SEQ_NET, self).__init__()
        self.drop_prob = drop_prob

        # FC layers
        self.fc1 = nn.Linear(384 + 384 + 512, 512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.KAF1 = KAF(512, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc2 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.KAF2 = KAF(256, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc3 = nn.Linear(256, 128)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.KAF3 = KAF(128, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc4 = nn.Linear(128, 2)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, uncovered, str_emb, seq_emb):
        num_res = len(uncovered)
        p1 = str_emb[uncovered]
        p1_info = torch.mean(str_emb, 0)
        p1_info = (p1_info.expand(num_res, 384)).float()
        partner_info = (seq_emb.expand(num_res, 512)).float()
        embedding = torch.cat((p1, p1_info, partner_info), 1)

        embedding = self.dropout(self.KAF1(self.fc1(embedding)))
        embedding = self.dropout(self.KAF2(self.fc2(embedding)))
        embedding = self.dropout(self.KAF3(self.fc3(embedding)))
        embedding = self.fc4(embedding)
        return embedding

class SEQ_STR_NET1(nn.Module):
    def __init__(self, unit, input_dim):
        super(SEQ_STR_NET1, self).__init__()
        self.output_size = 2
        self.n_layers = 2
        self.hidden_dim = 256
        self.drop_prob = 0.5

        if unit == 'LSTM':
            self.rnn_unit = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = self.drop_prob, bidirectional = True)
        if unit == 'GRU':
            self.rnn_unit = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = self.drop_prob, bidirectional = True)
        for weight in self.rnn_unit.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.xavier_uniform_(weight)

        self.dropout = nn.Dropout(self.drop_prob)
        self.fc1 = nn.Linear(self.hidden_dim * 6, 512)
        self.KAF1 = KAF(512, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(512, 256)
        self.KAF2 = KAF(256, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 128)
        self.KAF3 = KAF(128, D = 20, boundary = 120, kernel = 'gaussian')
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(128, self.output_size)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, unit, p1, hidden):
        p1 = p1.float()
        p1_out, p1_hidden = self.rnn_unit(p1, hidden)
        p1_out = p1_out.squeeze()

        if unit == 'LSTM':
            p1_hidden = p1_hidden[0]

        return p1_out

    def init_hidden(self, unit, batch_size):
        weight = next(self.parameters()).data
        if unit == 'LSTM':
            hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        if unit == 'GRU':
            hidden = weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_()
        return hidden

class SEQ_STR_NET2(nn.Module):
    def __init__(self, gcn_input_dim, rnn_input_dim):
        super(SEQ_STR_NET2, self).__init__()
        self.gcn_input_dim = gcn_input_dim
        self.gcn_output_dim = 128
        self.gcn_layers = 2
        self.gcn_drop_prob = 0
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = 128
        self.rnn_layers = 2
        self.rnn_drop_prob = 0.15671715
        self.rnn_batch_size = 1
        self.drop_prob = 0.48600144
        self.output_size = 2

        # RNN
        self.rnn_unit = nn.GRU(self.rnn_input_dim, self.rnn_hidden_dim, self.rnn_layers, batch_first = True, dropout = self.rnn_drop_prob, bidirectional = True)
        for weight in self.rnn_unit.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.xavier_uniform_(weight)

        # GCN layers
        self.gcnLayers = ARMAConvLayers(self.gcn_input_dim, self.gcn_output_dim, self.gcn_layers, self.gcn_drop_prob, D = 20, boundary = 150, kernel = 'gaussian')

        # FC layers
        self.fc1 = nn.Linear((self.gcn_output_dim + self.rnn_hidden_dim * 2) * 3, 512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.KAF1 = KAF(512, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc2 = nn.Linear(512, 512)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.KAF2 = KAF(512, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc3 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.KAF3 = KAF(256, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc4 = nn.Linear(256, self.output_size)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, gcn_feat, edge_index, rnn_feat, h, res_id):
        gcn_out = self.gcnLayers(gcn_feat, edge_index)
        rnn_feat = rnn_feat.float()
        rnn_out, hidden = self.rnn_unit(rnn_feat, h)
        rnn_out = rnn_out.squeeze()
        rnn_out = rnn_out[res_id]
        emb = torch.cat((gcn_out, rnn_out), 1)
        emb = torch.mean(emb, 0)
        return emb

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_layers * 2, self.rnn_batch_size, self.rnn_hidden_dim).zero_()
        return hidden

class SEQ_STR_NET(nn.Module):
    def __init__(self, drop_prob):
        super(SEQ_STR_NET, self).__init__()
        self.drop_prob = drop_prob

        # FC layers
        self.fc1 = nn.Linear(512 + 512 + 384, 512)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.KAF1 = KAF(512, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc2 = nn.Linear(512, 512)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.KAF2 = KAF(512, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc3 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.KAF3 = KAF(256, D = 20, boundary = 150, kernel = 'gaussian')
        self.fc4 = nn.Linear(256, 2)
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, res_samples, seq_emb, str_emb):
        num_res = len(res_samples)
        p1 = seq_emb[res_samples]
        p1_info = torch.mean(seq_emb, 0)
        p1_info = (p1_info.expand(num_res, 512)).float()
        partner_info = (str_emb.expand(num_res, 384)).float()
        embedding = torch.cat((p1, p1_info, partner_info), 1)

        embedding = self.dropout(self.KAF1(self.fc1(embedding)))
        embedding = self.dropout(self.KAF2(self.fc2(embedding)))
        embedding = self.dropout(self.KAF3(self.fc3(embedding)))
        embedding = self.fc4(embedding)
        return embedding
