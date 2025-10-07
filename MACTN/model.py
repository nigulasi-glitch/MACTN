# 导入必要的PyTorch和PyTorch Geometric模块
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import dropout_adj as dropout_edge
import random
import copy

EPS = 1e-15


def dropout_feat(x, drop_prob):

    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class EMA():

    def __init__(self, beta):

        super().__init__()
        self.beta = beta

    def update_average(self, old, new):

        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):

    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):

    for p in model.parameters():
        p.requires_grad = val


class Encoder(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, activation=torch.relu,
                 base_model=GCNConv, num_layers: int = 1):

        super(Encoder, self).__init__()
        self.base_model = base_model
        self.activation = activation
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(base_model(in_dim, hid_dim))
        for _ in range(num_layers - 1):
            self.convs.append(base_model(hid_dim, hid_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return x


class FusionModule(nn.Module):
    def __init__(self, hid_dim):
        super(FusionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 3),
            nn.Softmax(dim=-1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.414)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, z0, z1, z2):
        combined = torch.cat([z0.unsqueeze(1), z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        weights = self.attention(combined.view(combined.size(0), -1))
        weighted_combined = (combined * weights.unsqueeze(-1)).sum(dim=1)
        return weighted_combined


class CTHNE(nn.Module):
    def __init__(self,
                 encoder,
                 hid_dim,
                 num_relations,
                 tau: float = 0.2,
                 pe: float = 0.2,
                 pf: float = 0.2,
                 alpha: float = 0.5,
                 moving_average_decay: float = 0.2):

        super(CTHNE, self).__init__()

        self.loss_weights = nn.Parameter(torch.ones(4), requires_grad=True)
        self.fusion_module = FusionModule(hid_dim)

        self.online_encoder = encoder
        self.target_encoder1 = copy.deepcopy(self.online_encoder)
        self.target_encoder2 = copy.deepcopy(self.online_encoder)

        self.adjs_l_edge_indices = None

        set_requires_grad(self.target_encoder1, False)
        set_requires_grad(self.target_encoder2, False)

        self.target_ema_updater = EMA(moving_average_decay)

        self.hid_dim = hid_dim
        self.pe = pe
        self.pf = pf
        self.num_relations = num_relations
        self.tau = tau
        self.alpha = alpha

        self.local_projector = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, hid_dim))
        self.global_projector = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.PReLU(), nn.Linear(hid_dim, hid_dim))
        self.weight = nn.Parameter(torch.Tensor(hid_dim, hid_dim), requires_grad=True)

        self.reset_parameters()

    def reset_moving_average(self):
        del self.target_encoder1
        del self.target_encoder2
        self.target_encoder1 = None
        self.target_encoder2 = None

    def update_ma(self):
        assert self.target_encoder1 or self.target_encoder2 is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder1, self.online_encoder)
        update_moving_average(self.target_ema_updater,
                              self.target_encoder2, self.online_encoder)

    def reset_parameters(self):

        nn.init.xavier_normal_(self.weight, gain=1.414)
        self.online_encoder.reset_parameters()

        for model in self.local_projector:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.global_projector:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def forward(self, x_0, x_1, x_2, edge_indices, combine):

        zs = []
        for edge_index in edge_indices:
            if x_0 is not None:
                z0 = self.online_encoder(x_0, edge_index)
            else:
                if x_1 is not None:
                    z0 = torch.zeros(x_1.size(0), self.hid_dim, device=x_1.device)
                elif x_2 is not None:
                    z0 = torch.zeros(x_2.size(0), self.hid_dim, device=x_2.device)
                else:
                    raise ValueError("At least one of x_0, x_1, x_2 must be provided")

            if x_1 is not None:
                z1 = self.online_encoder(x_1, edge_index)
            else:
                if x_0 is not None:
                    z1 = torch.zeros(x_0.size(0), self.hid_dim, device=x_0.device)
                elif x_2 is not None:
                    z1 = torch.zeros(x_2.size(0), self.hid_dim, device=x_2.device)
                else:
                    raise ValueError("At least one of x_0, x_1, x_2 must be provided")

            if x_2 is not None:
                z2 = self.online_encoder(x_2, edge_index)
            else:
                if x_0 is not None:
                    z2 = torch.zeros(x_0.size(0), self.hid_dim, device=x_0.device)
                elif x_1 is not None:
                    z2 = torch.zeros(x_1.size(0), self.hid_dim, device=x_1.device)
                else:
                    raise ValueError("At least one of x_0, x_1, x_2 must be provided")

            z_combined = self.fusion_module(z0, z1, z2)
            zs.append(z_combined)

        if combine == 'concat':
            embeddings = torch.concat(zs, dim=-1)
        elif combine == 'mean':
            embeddings = torch.stack(zs).mean(dim=0)
        else:
            raise TypeError('Unsupported fuse function!')

        return embeddings

    def loss(self, x_0, x_1, x_2):
        loss = 0.
        num_contrasts = 0

        if self.adjs_l_edge_indices is None or self.adjs_h_edge_indices is None:
            raise ValueError(
                "adjs_l_edge_indices or adjs_h_edge_indices is not set. Please provide them before training.")

        min_length = min(len(self.adjs_l_edge_indices), len(self.adjs_h_edge_indices), self.num_relations)

        for i in range(min_length):
            for j in range(i, min_length):
                adj_l_1 = self.adjs_l_edge_indices[i]
                adj_l_2 = self.adjs_l_edge_indices[j]
                adj_h_1 = self.adjs_h_edge_indices[i]
                adj_h_2 = self.adjs_h_edge_indices[j]

                loss += self.contrast(x_0, x_1, x_2, adj_l_1, adj_l_2, adj_h_1, adj_h_2)
                num_contrasts += 1

        return loss / num_contrasts if num_contrasts > 0 else 0

    def contrast(self, x_0, x_1, x_2, adj_l_1, adj_l_2, adj_h_1, adj_h_2):
        edge_index_0 = random.choice([adj_l_1, adj_l_2])

        edge_index_1 = adj_l_1
        edge_index_2 = adj_h_1

        z0 = self.online_encoder(x_0, edge_index_0)
        z1 = self.online_encoder(x_1, edge_index_1)
        z2 = self.online_encoder(x_2, edge_index_2)

        with torch.no_grad():
            target_proj_01 = self.target_encoder1(x_0, edge_index_0)
            target_proj_11 = self.target_encoder1(x_1, edge_index_1)
            target_proj_02 = self.target_encoder2(x_0, edge_index_0)
            target_proj_22 = self.target_encoder2(x_2, edge_index_2)

        structure_sim = (z1 @ z1.T) + (target_proj_01 @ target_proj_01.T) + (target_proj_22 @ target_proj_22.T)
        structure_sim = F.normalize(structure_sim)

        adj_dense = torch.sparse_coo_tensor(
            edge_index_0,
            torch.ones(edge_index_0.size(1)),
            torch.Size([x_0.size(0), x_0.size(0)])
        ).to_dense()

        recon_structure_loss = torch.mean((adj_dense - structure_sim) ** 2)

        l_cn_1 = self.alpha * self.compute_loss(z1, target_proj_01.detach(), loss_type='agg') + \
                 (1.0 - self.alpha) * \
                 self.compute_loss(z1, target_proj_22.detach(), loss_type='agg')

        l_cn_2 = self.alpha * self.compute_loss(z2, target_proj_11.detach(), loss_type='agg') + \
                 (1.0 - self.alpha) * \
                 self.compute_loss(z2, target_proj_02.detach(), loss_type='agg')

        l_cn = (l_cn_1 + l_cn_2) / 2

        l_cv_0 = self.compute_loss(z0, z1, loss_type='inter')
        l_cv_1 = self.compute_loss(z1, z2, loss_type='inter')
        l_cv_2 = self.compute_loss(z2, z0, loss_type='inter')
        l_cv = (l_cv_0 + l_cv_1 + l_cv_2) / 3

        l_gl = (self.global_loss(z1, z2) + self.global_loss(z2, z1)) / 2

        log_weights = F.softmax(self.loss_weights, dim=0)
        loss = log_weights[0] * recon_structure_loss + \
               log_weights[1] * l_cn + \
               log_weights[2] * l_cv + \
               log_weights[3] * l_gl

        return loss

    def _sim(self, z1: torch.Tensor, z2: torch.Tensor):

        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _similarity(self, z1, z2):

        return torch.exp(self._sim(z1, z2) / self.tau)

    def agg_loss(self, z1, z2):

        sim_matrix = self._similarity(z1, z2)
        diag_sim = sim_matrix.diag()

        loss = -torch.log(-torch.log(diag_sim / sim_matrix.sum(dim=-1)))
        return loss.mean()

    def inter_loss(self, z1, z2):

        sim_intra = self._similarity(z1, z1)
        sim_inter = self._similarity(z1, z2)

        diag_sim = (sim_inter).diag()
        loss = -torch.log(diag_sim /
                          (sim_intra.sum(dim=-1) +
                           sim_inter.sum(dim=-1) -
                           (sim_intra).diag()))
        return loss.mean()

    def compute_loss(self, z1, z2, loss_type='agg'):

        h1, h2 = self.local_projector(z1), self.local_projector(z2)

        if loss_type == 'agg':
            return self.agg_loss(h1, h2)
        elif loss_type == 'inter':
            return self.inter_loss(h1, h2)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def readout(self, z):

        return z.mean(dim=0)

    def discriminate(self, z, summary, sigmoid=True):

        summary = torch.matmul(self.weight, summary)
        value = torch.matmul(z, summary)
        return torch.sigmoid(value) if sigmoid is True else value

    def global_loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor):

        s = self.readout(pos_z)
        h = self.global_projector(s)

        pos_loss = -torch.log(self.discriminate(pos_z, h, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, h, sigmoid=True) + EPS).mean()
        loss = (pos_loss + neg_loss) * 0.5

        return loss