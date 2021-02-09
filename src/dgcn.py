import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from gcn import BaseGNNNet
from krnn import MyKRNNEncoder
from n_beats import NBeatsEncoder


class GCNConv(PyG.MessagePassing):
    def __init__(self, gcn_in_dim, config, **kwargs):
        super().__init__(aggr=config.gcn_aggr, node_dim=-2, **kwargs)

        self.gcn_in_dim = gcn_in_dim
        self.gcn_node_dim = config.gcn_node_dim
        self.gcn_dim = config.gcn_dim
        self.gcn_dropout = config.gcn_dropout

        # TODO: consider adding LayerNorm here?
        self.fea_map = nn.Sequential(
            nn.Linear(self.gcn_in_dim+self.gcn_node_dim, self.gcn_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.gcn_dropout),
            nn.Linear(self.gcn_dim, self.gcn_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.gcn_dropout),
        )

    def forward(self, x, edge_index, edge_weight, node_type):
        batch_size, num_nodes, _ = x.shape
        assert num_nodes == node_type.shape[0]
        node_dim = node_type.shape[1]
        if edge_weight.shape[0] == x.shape[0]:
            num_edges = edge_weight.shape[1]
            edge_weight = edge_weight.reshape(batch_size, num_edges, 1)
        else:
            num_edges = edge_weight.shape[0]
            edge_weight = edge_weight.reshape(1, num_edges, 1).expand(batch_size, -1, -1)
        node_type = node_type.reshape(1, num_nodes, node_dim).expand(batch_size, -1, -1)

        # Calculate type-aware node info
        if isinstance(x, torch.Tensor):
            x_in = self.fea_map(torch.cat([x, node_type], dim=-1))
            x_out = x
        else:
            x_in = self.fea_map(torch.cat([x[0], node_type], dim=-1))
            x_out = x[1]

        aggr_out = self.propagate(edge_index, x=x_in, edge_weight=edge_weight, node_type=node_type)

        return x_out + aggr_out

    def message(self, x_j, edge_weight):
        return x_j * edge_weight


class GCNNet(BaseGNNNet):
    def __init__(self, gcn_in_dim, config):
        super().__init__()

        self.num_node_types = config.num_node_types
        self.node_dim = config.gcn_node_dim
        self.layer_num = config.gcn_layer_num
        assert self.layer_num >= 1

        self.node_type_emb = nn.Embedding(self.num_node_types, self.node_dim)

        convs = [GCNConv(gcn_in_dim, config)]
        for _ in range(self.layer_num-1):
            convs.append(GCNConv(config.gcn_dim, config))
        self.convs = nn.ModuleList(convs)

    def subgraph_forward(self, x, g, edge_weight=None):
        edge_index = g['edge_index']
        if edge_weight is None:
            # edge_weight in arguments has the highest priority
            edge_weight = g['edge_attr']
        node_type = self.node_type_emb(g['node_type'])

        for conv in self.convs:
            # conv already implements the residual connection
            x = conv(x, edge_index, edge_weight, node_type)

        return x


class EdgeNet(GCNNet):
    def __init__(self, gcn_in_dim, config):
        super().__init__(gcn_in_dim, config)
        self.gcn_dim = config.gcn_dim
        self.gcn_dropout = config.gcn_dropout

        self.fea_map = nn.Sequential(
            nn.Linear(2*self.gcn_dim, self.gcn_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.gcn_dropout),
            nn.Linear(self.gcn_dim, 1),
            nn.ReLU(),
        )

    def subgraph_forward(self, x, g):
        x = super().subgraph_forward(x, g)
        edge_index = g['edge_index'].permute(1, 0)  # [num_edges, 2]
        edge_x = x[:, edge_index, :].reshape(x.shape[0], edge_index.shape[0], self.gcn_dim*2)
        edge_pred = self.fea_map(edge_x)

        return edge_pred


class DGCNModel(nn.Module):
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

        self.edge_gcn = EdgeNet(self.rnn_hid_dim, config)
        self.gcn = GCNNet(self.rnn_hid_dim, config)
        self.gcn_fc = nn.Linear(config.gcn_dim, config.lookahead_days)

        self.edge_gate = None
        self.y_rnn = None
        self.y_gcn = None

    def forward(self, input_day, g):
        # rnn_out.size: [batch_size, node_num, hidden_dim]
        # y_rnn.size: [batch_size, node_num, forecast_len]
        if self.rnn_type == 'krnn':
            # kr_out.size: [batch_size, node_num, seq_len, hidden_dim]
            kr_out = self.rnn(input_day, g)
            rnn_out, _ = kr_out.max(dim=-2)
            self.y_rnn = self.rnn_fc(rnn_out)
        elif self.rnn_type == 'nbeats':
            # nb_out.size: [batch_size, node_num, hidden_dim, seq_len]
            nb_out, self.y_rnn = self.rnn(input_day, g)
            rnn_out, _ = nb_out.max(dim=-1)
        else:
            raise Exception(f'Unsupported rnn type {self.rnn_type}')

        # edge_gate.size: [batch_size, edge_num, 1]
        # gcn_out.size: [batch_size, node_num, hidden_dim]
        # y_gcn.size: [batch_size, node_num, forecast_len]
        self.edge_gate = self.edge_gcn(rnn_out, g)
        gcn_out = self.gcn(rnn_out, g, edge_weight=self.edge_gate)
        self.y_gcn = self.gcn_fc(gcn_out)

        y = self.y_rnn + self.y_gcn

        return y