import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GCNNet, GateGCNNet
from krnn import MyKRNNEncoder
from n_beats import NBeatsEncoder


class TGCNModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.rnn_type = config.rnn_type
        if self.rnn_type == 'krnn':
            self.rnn = MyKRNNEncoder(config)
            self.rnn_hid_dim = config.rnn_dim
            self.rnn_fc = nn.Linear(self.rnn_hid_dim, config.lookahead_days)
        elif self.rnn_type == 'nbeats':
            self.rnn = NBeatsEncoder(config)
            self.rnn_hid_dim = config.hidden_dim
        else:
            raise Exception(f'Unsupported rnn type {self.rnn_type}')

        if config.gcn_type == 'gcn':
            self.gcn = GCNNet(self.rnn_hid_dim, config.gcn_dim,
                              config.num_nodes, config.num_node_types, config.num_edge_types,
                              aggr=config.gcn_aggr, layer_num=config.gcn_layer_num)
        elif config.gcn_type == 'gate_gcn':
            self.gcn = GateGCNNet(self.rnn_hid_dim, config.gcn_dim,
                                  config.num_node_types, config.num_edge_types,
                                  config.gcn_node_dim, config.gcn_edge_dim,
                                  aggr=config.gcn_aggr, layer_num=config.gcn_layer_num)
        else:
            raise Exception(f'Unsupported gcn_type: {config.gcn_type}')
        self.gcn_fc = nn.Linear(config.gcn_dim, config.lookahead_days)

        self.coef_fc = nn.Sequential(
            nn.Linear(config.gcn_dim, config.lookahead_days),
            nn.Sigmoid(),
        )
        self.rnn_coef = None

    def forward(self, input_day, g):
        # rnn_out.size: [batch_size, node_num, hidden_dim]
        # y_rnn.size: [batch_size, node_num, forecast_len]
        if self.rnn_type == 'krnn':
            # kr_out.size: [batch_size, node_num, seq_len, hidden_dim]
            kr_out = self.rnn(input_day, g)
            rnn_out, _ = kr_out.max(dim=-2)
            y_rnn = self.rnn_fc(rnn_out)
        elif self.rnn_type == 'nbeats':
            # nb_out.size: [batch_size, node_num, hidden_dim, seq_len]
            nb_out, y_rnn = self.rnn(input_day, g)
            rnn_out, _ = nb_out.max(dim=-1)
        else:
            raise Exception(f'Unsupported rnn type {self.rnn_type}')

        # gcn_out.size: [batch_size, node_num, hidden_dim]
        # y_gcn.size: [batch_size, node_num, forecast_len]
        gcn_out = self.gcn(rnn_out, g)
        y_gcn = self.gcn_fc(gcn_out)

        self.rnn_coef = self.coef_fc(gcn_out)
        y = y_rnn * self.rnn_coef + y_gcn * (1 - self.rnn_coef)

        return y