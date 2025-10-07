import numpy as np
import scipy.sparse as sp
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.utils import degree, to_undirected
import warnings
from datasets import ACM, AMiner, FreeBase, get_dataset
warnings.filterwarnings('ignore')

class SimFeatureEnhanceWithDenoising(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, str_enc_dim=16, thre=0.9):
        super(SimFeatureEnhanceWithDenoising, self).__init__()
        self.w = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.linear = torch.nn.Linear(in_channels + str_enc_dim, in_channels)
        self.thre = thre
        self.num_w = out_channels
        self.denoise_weight = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w)
        nn.init.constant_(self.denoise_weight, 0.5)

    def attention_(self, x):
        score = torch.mm(x, self.w.t())
        return score

    def augment(self, x: Tensor, rw_embeddings, epr=None):
        if not any(dim == 0 for dim in rw_embeddings.shape):
            x_hat = torch.cat([x, rw_embeddings], dim=1)
            x_0 = self.linear(x_hat)
        else:
            x_0 = x

        score = self.attention_(x_0)
        weight = F.softmax(score, dim=-1)

        if epr is not None:
            denoise_factor = 1 - self.denoise_weight * epr.unsqueeze(1)
            weight = weight * denoise_factor

        if self.num_w != 1:
            thresh = getMatrix(weight, self.thre).data
            weight = torch.threshold(weight, thresh, 0)

        p = weight.mm(self.w)
        return torch.add(x, p), p

def getMatrix(matrix, pr):
    non_zero_values = matrix[matrix != 0]
    num_ones = int(pr * non_zero_values.numel())
    sorted_values, _ = torch.sort(non_zero_values, descending=True)
    threshold_value = sorted_values[num_ones - 1]
    return threshold_value

def calculate_epr(data, target_node_type, metapath_adj):
    labels = data[target_node_type].y
    edge_index = metapath_adj.coalesce().indices()
    edge_index = to_undirected(edge_index)
    deg = degree(edge_index[0]).to(torch.int32)
    computed_deg = 0
    epr_list = []

    for i in range(labels.size(0)):
        d = deg[i]
        if d == 0:
            epr_list.append(0.0)
            continue
        neighbour = edge_index[1][computed_deg: computed_deg + d]
        computed_deg += d
        neighbour_deg = deg[neighbour]
        neighbour_y = labels[neighbour]
        wrong_neighbour_mask = (neighbour_y != labels[i])
        wrong_passing = 1 / neighbour_deg[wrong_neighbour_mask] ** 0.5
        all_passing = 1 / neighbour_deg ** 0.5
        if all_passing.sum() > 0:
            epr_value = (wrong_passing.sum() / all_passing.sum()).item()
        else:
            epr_value = 0.0
        epr_list.append(epr_value)

    return torch.tensor(epr_list, dtype=torch.float32)

def generate_random_walk_embeddings(adj, num_walks=10, walk_length=5, embedding_dim=16):
    adj_dense = adj.to_dense()
    num_nodes = adj_dense.shape[0]
    embeddings = torch.zeros(num_nodes, embedding_dim)

    for node in range(num_nodes):
        for walk in range(num_walks):
            current_node = node
            walk_embedding = torch.zeros(embedding_dim)

            for step in range(walk_length):
                neighbors = torch.nonzero(adj_dense[current_node]).flatten()
                if len(neighbors) > 0:
                    next_node = neighbors[torch.randint(0, len(neighbors), (1,))]
                    walk_embedding += torch.randn(embedding_dim) * 0.1
                    current_node = next_node.item()

            embeddings[node] += walk_embedding / walk_length
        embeddings[node] /= num_walks

    return embeddings

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    if sp.issparse(features):
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features.todense()
    else:
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features = r_mat_inv.dot(features)
        return features

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def get_node_count(data, node_type):
    if hasattr(data[node_type], 'num_nodes') and data[node_type].num_nodes is not None:
        return data[node_type].num_nodes
    if hasattr(data[node_type], 'x') and data[node_type].x is not None:
        return data[node_type].x.shape[0]
    max_index = -1
    for edge_type in data.edge_types:
        if node_type in edge_type:
            edge_index = data[edge_type].edge_index
            if edge_index is not None and edge_index.numel() > 0:
                max_index = max(max_index, edge_index.max().item())
    if max_index >= 0:
        return max_index + 1
    print(f"警告: 无法确定节点类型 '{node_type}' 的节点数量")
    return 1000

def build_metapath_adjacency(data, metapath, target_node_type):
    num_nodes = get_node_count(data, target_node_type)
    adj = sp.eye(num_nodes)
    for edge_type in metapath:
        src_type, dst_type = edge_type
        edge_index = data[src_type, dst_type].edge_index.numpy()
        num_src = get_node_count(data, src_type)
        num_dst = get_node_count(data, dst_type)
        rel_adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(num_src, num_dst)
        )
        adj = adj.dot(rel_adj)
    adj.data = np.ones_like(adj.data)
    return adj

def process_dataset_with_denoising_enhancement(dataset_name, enhance_dim=16):
    dataset, metapaths, target_node_type = get_dataset(dataset_name)
    data = dataset[0]
    feat = data[target_node_type].x.numpy()
    labels = data[target_node_type].y.numpy()
    feat = preprocess_features(feat)
    feat = th.FloatTensor(feat)
    label = encode_onehot(labels)
    label = th.FloatTensor(label)
    adjs = []
    for metapath in metapaths:
        adj = build_metapath_adjacency(data, metapath, target_node_type)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adjs.append(adj)
    if len(adjs) > 0:
        epr = calculate_epr(data, target_node_type, adjs[0])
    else:
        num_nodes = get_node_count(data, target_node_type)
        epr = torch.zeros(num_nodes)
    if len(adjs) > 0:
        rw_embeddings = generate_random_walk_embeddings(adjs[0], embedding_dim=enhance_dim)
    else:
        num_nodes = get_node_count(data, target_node_type)
        rw_embeddings = torch.zeros(num_nodes, enhance_dim)
    enhance_module = SimFeatureEnhanceWithDenoising(
        in_channels=feat.shape[1],
        out_channels=feat.shape[1],
        str_enc_dim=enhance_dim
    )
    enhanced_feat, _ = enhance_module.augment(feat, rw_embeddings, epr)
    print(f"数据集 '{dataset_name}' 的平均EPR: {epr.mean().item()}")
    return enhanced_feat, adjs, label