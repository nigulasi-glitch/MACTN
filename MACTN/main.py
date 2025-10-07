import warnings

from torch import nn

warnings.filterwarnings("ignore")

import argparse
import os.path as osp
from pathlib import Path
import shutil
import itertools
from collections import defaultdict

import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

import torch
import torch_geometric.transforms as T

from model import CTHNE, Encoder
from utils import set_random_seed, add_self_loop
from datasets import get_dataset
from augment.structure_augment import graph_process


def get_arguments():
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--dataset', type=str, default='aminer')
    parser.add_argument("--train_target", type=str, choices=["micro_f1", "nmi"], default="micro_f1",
                        help="Training target to optimize ('micro_f1' or 'nmi')")

    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--clf_runs', type=int, default=10)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--edge_drop_rate', type=float, default=0.2)
    parser.add_argument('--feature_drop_rate', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--moving_average_decay', type=float, default=0.)

    parser.add_argument('--train_splits', type=float, nargs='+', default=[0.2])
    parser.add_argument('--combine', type=str, default='concat')

    parser.add_argument('--graph_k', type=int, default=15)
    parser.add_argument('--k_pos', type=int, default=10)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return vars(args)


def train(model, x_0, x_1, x_2, optimizer):
    model.train()
    optimizer.zero_grad()

    loss = model.loss(x_0, x_1, x_2)

    loss.backward(retain_graph=True)
    optimizer.step()
    model.update_ma()

    return loss.item()


def test(embeddings, labels, train_split=0.2, runs=10):
    macro_f1_list = list()
    micro_f1_list = list()
    nmi_list = list()
    ari_list = list()

    for i in range(runs):
        x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, train_size=train_split, random_state=i)

        clf = SVC(probability=True)

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)

    for i in range(runs):
        kmeans = KMeans(n_clusters=len(torch.unique(labels)), algorithm='full')
        y_kmeans = kmeans.fit_predict(embeddings)

        nmi = normalized_mutual_info_score(labels, y_kmeans)
        ari = adjusted_rand_score(labels, y_kmeans)
        nmi_list.append(nmi)
        ari_list.append(ari)

    macro_f1 = np.array(macro_f1_list).max()
    micro_f1 = np.array(micro_f1_list).max()
    nmi = np.array(nmi_list).max()
    ari = np.array(ari_list).max()

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'nmi': nmi,
        'ari': ari
    }


def main():
    params = get_arguments()
    set_random_seed(params['seed'])
    train_target = params['train_target']
    device = torch.device('cuda:{}'.format(params['gpu']) if torch.cuda.is_available() else 'cpu')

    checkpoints_path = f'checkpoints'
    try:
        shutil.rmtree(checkpoints_path)
    except:
        pass
    Path(checkpoints_path).mkdir(parents=True, exist_ok=False)

    dataset, metapaths, target = get_dataset(params['dataset'])
    data = dataset[0]
    num_relations = len(metapaths)
    num_nodes = data[target].y.shape[0]
    num_feat = data[target].x.shape[1]

    from augment.structure_load_data import process_dataset as structure_augment
    from augment.unfied_augment import process_dataset_with_denoising_enhancement as denoising_augment
    from data.Generate_ICI.ACM1 import get_dataset_semantic as semantic_augment

    feat_structure, adj_s, label_structure = structure_augment(params['dataset'])
    feat_denoising, adj, label_denoising = denoising_augment(params['dataset'])
    dataset_semantic, metapaths, target = semantic_augment(params['dataset'])
    data_semantic = dataset_semantic[0]
    feat_semantic = data_semantic[target].x

    x0 = feat_structure.to(device)
    x1 = feat_denoising.to(device)
    x2 = feat_semantic.to(device)

    metapath_data = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True)(data)
    metapath_data = add_self_loop(metapath_data, num_relations, num_nodes)

    edge_indices = [edge_index.to(device) for edge_index in metapath_data.edge_index_dict.values()]
    labels = metapath_data[target].y

    class ArgsObject:
        def __init__(self, params):
            self.graph_k = params['graph_k']
            self.k_pos = params['k_pos']

    args_obj = ArgsObject(params)

    adjs_l, adjs_h, adjs_o, pos = graph_process(adj_s, feat_structure, args_obj)

    def convert_adj_to_edge_index(adj_list, device):
        edge_indices = []
        for adj in adj_list:
            if not isinstance(adj, torch.Tensor):
                adj = torch.tensor(adj, dtype=torch.float32)

            if adj.is_sparse:
                adj_dense = adj.to_dense()
            else:
                adj_dense = adj

            adj_binary = (adj_dense > 0).float()

            edge_index = torch.nonzero(adj_binary, as_tuple=False).t().to(device)

            if edge_index.size(1) == 0:
                num_nodes = adj_dense.size(0)
                edge_index = torch.stack([
                    torch.arange(num_nodes, device=device),
                    torch.arange(num_nodes, device=device)
                ])

            edge_indices.append(edge_index)
        return edge_indices

    adjs_l_edge_indices = convert_adj_to_edge_index(adjs_l, device)
    adjs_h_edge_indices = convert_adj_to_edge_index(adjs_h, device)

    encoder = Encoder(in_dim=num_feat, hid_dim=params['hid_dim'], num_layers=params['num_layers'])
    model = CTHNE(encoder=encoder, hid_dim=params['hid_dim'], num_relations=num_relations,
                  tau=params['tau'], pe=params['edge_drop_rate'], pf=params['feature_drop_rate'],
                  alpha=params['alpha'], moving_average_decay=params['moving_average_decay']).to(device)

    model.adjs_l_edge_indices = adjs_l_edge_indices
    model.adjs_h_edge_indices = adjs_h_edge_indices

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_epoch = 0
    best_value = 0
    patience_cnt = 0

    for i in range(1, params['epochs'] + 1):
        loss = train(model, x0, x1, x2, optimizer)

        if i % params['eval_interval'] == 0:
            embeddings = model(x0, x1, x2, edge_indices, 'concat').detach().cpu().numpy()
            results = test(embeddings, labels, train_split=0.2, runs=params['clf_runs'])
            print(
                f'Epoch {i}: Macro-F1: {results["macro_f1"]:.4f} | Micro-F1: {results["micro_f1"]:.4f} | NMI: {results["nmi"]:.4f} | ARI: {results["ari"]:.4f}')

            if train_target == 'micro_f1':
                if results['micro_f1'] > best_value:
                    best_value = results['micro_f1']
                    best_epoch = i
                    patience_cnt = 0
                    torch.save(model.state_dict(), osp.join(checkpoints_path, f'{i}.pkl'))
                else:
                    patience_cnt += 1
            elif train_target == 'nmi':
                if results['nmi'] > best_value:
                    best_value = results['nmi']
                    best_epoch = i
                    patience_cnt = 0
                    torch.save(model.state_dict(), osp.join(checkpoints_path, f'{i}.pkl'))
                else:
                    patience_cnt += 1

            if patience_cnt >= params['patience']:
                print(f"Early stopping at epoch {i}")
                break

    model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{best_epoch}.pkl')))

    embeddings = model(x0, x1, x2, edge_indices, 'concat').detach().cpu().numpy()
    results = test(embeddings, labels, train_split=0.2, runs=params['clf_runs'])

    print(
        f'Best Epoch: {best_epoch} | Train Split: 0.2 | Macro-F1: {results["macro_f1"]:.4f} | Micro-F1: {results["micro_f1"]:.4f} | NMI: {results["nmi"]:.4f} | ARI: {results["ari"]:.4f}')

    shutil.rmtree(checkpoints_path)


if __name__ == '__main__':
    main()