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

        self.fea_map = nn.Linear(self.gcn_in_dim, self.gcn_dim)

    def forward(self, x, edge_index, edge_weight):
        batch_size, num_nodes, _ = x.shape
        if edge_weight.shape[0] == x.shape[0]:
            num_edges = edge_weight.shape[1]
            edge_weight = edge_weight.reshape(batch_size, num_edges, 1)
        else:
            num_edges = edge_weight.shape[0]
            edge_weight = edge_weight.reshape(1, num_edges, 1).expand(batch_size, -1, -1)

        # Calculate type-aware node info
        if isinstance(x, torch.Tensor):
            x = self.fea_map(x)
        else:
            x = (self.fea_map(x[0]), self.fea_map(x[1]))

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight

    def update(self, aggr_out, x):
        if not isinstance(x, torch.Tensor):
            x = x[1]

        return x + aggr_out


class MyGATConv(PyG.MessagePassing):
    def __init__(self, gcn_in_dim, config, **kwargs):
        super().__init__(aggr=config.gcn_aggr, node_dim=-2, **kwargs)

        self.gcn_in_dim = gcn_in_dim
        self.gcn_node_dim = config.gcn_node_dim
        self.gcn_dim = config.gcn_dim

        self.fea_map = nn.Linear(self.gcn_in_dim, self.gcn_dim)
        self.msg_map = nn.Linear(self.gcn_dim, self.gcn_dim)

    def forward(self, x, edge_index, edge_weight):
        batch_size, num_nodes, _ = x.shape
        if edge_weight.shape[0] == x.shape[0]:
            num_edges = edge_weight.shape[1]
            edge_weight = edge_weight.reshape(batch_size, num_edges, 1)
        else:
            num_edges = edge_weight.shape[0]
            edge_weight = edge_weight.reshape(1, num_edges, 1).expand(batch_size, -1, -1)

        # Calculate type-aware node info
        if isinstance(x, torch.Tensor):
            x = self.fea_map(x)
        else:
            x = (self.fea_map(x[0]), self.fea_map(x[1]))

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        x_i = self.msg_map(x_i)
        x_j = self.msg_map(x_j)
        gate = torch.sigmoid((x_i * x_j).sum(dim=-1, keepdim=True))
        return x_j * edge_weight * gate

    def update(self, aggr_out, x):
        if not isinstance(x, torch.Tensor):
            x = x[1]

        return x + aggr_out


class GCNNet(BaseGNNNet):
    def __init__(self, gcn_in_dim, config):
        super().__init__()

        self.layer_num = config.gcn_layer_num
        assert self.layer_num >= 1

        if config.gcn_type == 'gcn':
            GCNClass = GCNConv
        elif config.gcn_type == 'gat':
            GCNClass = MyGATConv
        else:
            raise Exception(f'Unsupported gcn_type {config.gcn_type}')

        convs = [GCNClass(gcn_in_dim, config)]
        for _ in range(self.layer_num-1):
            convs.append(GCNClass(config.gcn_dim, config))
        self.convs = nn.ModuleList(convs)

    def subgraph_forward(self, x, g, edge_weight=None):
        edge_index = g['edge_index']
        if edge_weight is None:
            # edge_weight in arguments has the highest priority
            edge_weight = g['edge_attr']

        for conv in self.convs:
            # conv already implements the residual connection
            x = conv(x, edge_index, edge_weight)

        return x


class EdgeNet(GCNNet):
    def __init__(self, gcn_in_dim, config):
        super().__init__(gcn_in_dim, config)

        self.gcn_dim = config.gcn_dim
        self.num_nodes = config.num_nodes
        self.node_dim = config.gcn_node_dim
        self.edge_dim = 2*(self.gcn_dim+self.node_dim)

        self.node_emb = nn.Embedding(self.num_nodes, self.node_dim)
        self.edge_map = nn.Sequential(
            nn.Linear(self.edge_dim, 1),
            nn.ReLU(),
        )

    def subgraph_forward(self, x, g):
        x = super().subgraph_forward(x, g)
        batch_size, node_num, _ = x.shape

        # add node-specific representations
        n_id = g['cent_n_id']
        x_id = self.node_emb(n_id)\
            .reshape(1, node_num, self.node_dim)\
            .expand(batch_size, -1, -1)
        x = torch.cat([x, x_id], dim=-1)

        # calculate the edge gate for each node pair
        edge_index = g['edge_index'].permute(1, 0)  # [num_edges, 2]
        edge_num = edge_index.shape[0]
        edge_x = x[:, edge_index, :]\
            .reshape(batch_size, edge_num, self.edge_dim)
        edge_gate = self.edge_map(edge_x)

        return edge_gate


class ZGCNModel(nn.Module):
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

        if config.use_default_edge:
            self.edge_gcn = None
        else:
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
        if self.edge_gcn is None:
            gcn_out = self.gcn(rnn_out, g)
        else:
            self.edge_gate = self.edge_gcn(rnn_out, g)
            gcn_out = self.gcn(rnn_out, g, edge_weight=self.edge_gate)
        self.y_gcn = self.gcn_fc(gcn_out)

        y = self.y_rnn + self.y_gcn

        return y