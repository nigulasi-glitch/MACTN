import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
from datasets import ACM, AMiner, FreeBase, get_dataset
from augment.structure_augment import graph_process
from params import set_params
warnings.filterwarnings('ignore')
args = set_params()

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

    return 1000

def build_metapath_adjacency(data, metapath, target_node_type):
    num_nodes = get_node_count(data, target_node_type)
    adj = sp.eye(num_nodes)

    for i, edge_type in enumerate(metapath):
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

def process_dataset(dataset_name):
    dataset, metapaths, target_node_type = get_dataset(dataset_name)
    data = dataset[0]


    feat = data[target_node_type].x.numpy()
    labels = data[target_node_type].y.numpy()

    feat = preprocess_features(feat)
    feat = th.FloatTensor(feat)

    label = encode_onehot(labels)
    label = th.FloatTensor(label)

    adjs = []
    for i, metapath in enumerate(metapaths):
        print(f"构建元路径 {i + 1}: {metapath}")
        adj = build_metapath_adjacency(data, metapath, target_node_type)
        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adjs.append(adj)

    return feat, adjs, label