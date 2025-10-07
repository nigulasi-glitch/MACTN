import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import embedding
from args import make_args
import pickle
import os.path as osp
import warnings

warnings.filterwarnings('ignore')

class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat)
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape)
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)

        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu = nn.PReLU()
        x = prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class HGCL(nn.Module):
    def __init__(self, args, userNum, itemNum, userMat, itemMat, uiMat, hide_dim, Layers):
        super(HGCL, self).__init__()

        self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.uuMat = userMat
        self.iiMat = itemMat
        self.uiMat = uiMat
        self.hide_dim = hide_dim
        self.LayerNums = Layers

        uimat = self.uiMat[: self.userNum, self.userNum:]
        values = torch.FloatTensor(uimat.tocoo().data)
        indices = np.vstack((uimat.tocoo().row, uimat.tocoo().col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = uimat.tocoo().shape
        uimat1 = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.uiadj = uimat1
        self.iuadj = uimat1.transpose(0, 1)

        self.gating_weightub = nn.Parameter(torch.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightub.data)
        self.gating_weightu = nn.Parameter(torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        self.gating_weightib = nn.Parameter(torch.FloatTensor(1, hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti = nn.Parameter(torch.FloatTensor(hide_dim, hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)

        self.encoder = nn.ModuleList()
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())

        self.k = args.rank
        k = self.k
        self.mlp = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp1 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp2 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.mlp3 = MLP(hide_dim, hide_dim * k, hide_dim // 2, hide_dim * k)
        self.meta_netu = nn.Linear(hide_dim * 3, hide_dim, bias=True)
        self.meta_neti = nn.Linear(hide_dim * 3, hide_dim, bias=True)
        self.embedding_dict = nn.ModuleDict(
            {
                'uu_emb': torch.nn.Embedding(userNum, hide_dim),
                'ii_emb': torch.nn.Embedding(itemNum, hide_dim),
                'user_emb': torch.nn.Embedding(userNum, hide_dim),
                'item_emb': torch.nn.Embedding(itemNum, hide_dim),
            }
        )

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_normal_
        embedding_dict = nn.ParameterDict(
            {
                'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
                'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
            }
        )
        return embedding_dict

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def self_gatingu(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightu) + self.gating_weightub))

    def self_gatingi(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weighti) + self.gating_weightib))

    def metafortansform(self, auxiembedu, targetembedu, auxiembedi, targetembedi):
        uneighbor = torch.matmul(self.uiadj, self.ui_itemEmbedding)
        ineighbor = torch.matmul(self.iuadj, self.ui_userEmbedding)
        tembedu = (self.meta_netu(torch.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach()))
        tembedi = (self.meta_neti(torch.cat((auxiembedi, targetembedi, ineighbor), dim=1).detach()))
        metau1 = self.mlp(tembedu).reshape(-1, self.hide_dim, self.k)
        metau2 = self.mlp1(tembedu).reshape(-1, self.k, self.hide_dim)
        metai1 = self.mlp2(tembedi).reshape(-1, self.hide_dim, self.k)
        metai2 = self.mlp3(tembedi).reshape(-1, self.k, self.hide_dim)

        meta_biasu = (torch.mean(metau1, dim=0))
        meta_biasu1 = (torch.mean(metau2, dim=0))
        meta_biasi = (torch.mean(metai1, dim=0))
        meta_biasi1 = (torch.mean(metai2, dim=0))

        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)

        tembedus = (torch.sum(torch.multiply((auxiembedu).unsqueeze(-1), low_weightu1), dim=1))
        tembedus = torch.sum(torch.multiply((tembedus).unsqueeze(-1), low_weightu2), dim=1)
        tembedis = (torch.sum(torch.multiply((auxiembedi).unsqueeze(-1), low_weighti1), dim=1))
        tembedis = torch.sum(torch.multiply((tembedis).unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed

    def forward(self, norm=1):
        item_index = np.arange(0, self.itemNum)
        user_index = np.arange(0, self.userNum)
        ui_index = np.array(user_index.tolist() + [i + self.userNum for i in item_index])

        userembed0 = self.embedding_dict['user_emb'].weight
        itemembed0 = self.embedding_dict['item_emb'].weight
        uu_embed0 = self.self_gatingu(userembed0)
        ii_embed0 = self.self_gatingi(itemembed0)
        self.ui_embeddings = torch.cat([userembed0, itemembed0], 0)
        self.all_user_embeddings = [uu_embed0]
        self.all_item_embeddings = [ii_embed0]
        self.all_ui_embeddings = [self.ui_embeddings]

        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            if i == 0:
                userEmbeddings0 = layer(uu_embed0, self.uuMat, user_index)
                itemEmbeddings0 = layer(ii_embed0, self.iiMat, item_index)
                uiEmbeddings0 = layer(self.ui_embeddings, self.uiMat, ui_index)
            else:
                userEmbeddings0 = layer(userEmbeddings, self.uuMat, user_index)
                itemEmbeddings0 = layer(itemEmbeddings, self.iiMat, item_index)
                uiEmbeddings0 = layer(uiEmbeddings, self.uiMat, ui_index)

            self.ui_userEmbedding0, self.ui_itemEmbedding0 = torch.split(uiEmbeddings0, [self.userNum, self.itemNum])
            userEd = (userEmbeddings0 + self.ui_userEmbedding0) / 2
            itemEd = (itemEmbeddings0 + self.ui_itemEmbedding0) / 2
            userEmbeddings = userEd
            itemEmbeddings = itemEd
            uiEmbeddings = torch.cat([userEd, itemEd], 0)

            if norm == 1:
                norm_embeddings = F.normalize(userEmbeddings0, p=2, dim=1)
                self.all_user_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings += [norm_embeddings]
            else:
                self.all_user_embeddings += [userEmbeddings]
                self.all_item_embeddings += [itemEmbeddings]
                self.all_ui_embeddings += [uiEmbeddings]

        self.userEmbedding = torch.stack(self.all_user_embeddings, dim=1)
        self.userEmbedding = torch.mean(self.userEmbedding, dim=1)
        self.itemEmbedding = torch.stack(self.all_item_embeddings, dim=1)
        self.itemEmbedding = torch.mean(self.itemEmbedding, dim=1)
        self.uiEmbedding = torch.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding = torch.mean(self.uiEmbedding, dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = torch.split(self.uiEmbedding, [self.userNum, self.itemNum])

        metatsuembed, metatsiembed = self.metafortansform(
            self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding
        )
        self.userEmbedding = self.userEmbedding + metatsuembed
        self.itemEmbedding = self.itemEmbedding + metatsiembed
        return self.userEmbedding, self.itemEmbedding, (
                self.args.wu1 * self.ui_userEmbedding + self.args.wu2 * self.userEmbedding), (
                self.args.wi1 * self.ui_itemEmbedding + self.args.wi2 * self.itemEmbedding)

class Hope():
    def __init__(self, args, data, distanceMat, itemMat=None):
        self.args = args
        self.data = data

        print(f"Hope初始化: distanceMat类型 = {type(distanceMat)}")

        if isinstance(distanceMat, dict):
            print("使用字典格式的距离矩阵")
            self.userDistanceMat = distanceMat.get('movie_movie', None)
            self.itemDistanceMat = distanceMat.get('actor_actor', None)
            self.uiDistanceMat = distanceMat.get('movie_actor', None)

            if self.userDistanceMat is None:
                self.userDistanceMat = distanceMat.get('paper_paper', None)
            if self.itemDistanceMat is None:
                self.itemDistanceMat = distanceMat.get('author_author', None)
            if self.uiDistanceMat is None:
                self.uiDistanceMat = distanceMat.get('paper_author', None)

            print(
                f"提取的矩阵 - 用户: {type(self.userDistanceMat)}, 物品: {type(self.itemDistanceMat)}, UI: {type(self.uiDistanceMat)}")

        else:
            print("使用元组格式的距离矩阵")
            self.userDistanceMat, self.itemDistanceMat, self.uiDistanceMat = distanceMat

        if self.userDistanceMat is not None and hasattr(self.userDistanceMat, 'shape'):
            self.userMat = (self.userDistanceMat != 0) * 1
            print(f"用户矩阵形状: {self.userMat.shape}")
        else:
            print("警告: 用户距离矩阵无效，创建空矩阵")
            self.userMat = sp.csr_matrix((0, 0))

        if self.itemDistanceMat is not None and hasattr(self.itemDistanceMat, 'shape'):
            self.itemMat = (self.itemDistanceMat != 0) * 1
            print(f"物品矩阵形状: {self.itemMat.shape}")
        else:
            print("警告: 物品距离矩阵无效，创建空矩阵")
            self.itemMat = sp.csr_matrix((0, 0))

        if self.uiDistanceMat is not None and hasattr(self.uiDistanceMat, 'shape'):
            top = sp.hstack([self.userMat, self.uiDistanceMat])
            bottom = sp.hstack([self.uiDistanceMat.T, self.itemMat])
            self.uiMat = sp.vstack([top, bottom])
            self.uiMat = (self.uiMat != 0) * 1
            print(f"UI矩阵形状: {self.uiMat.shape}")
        else:
            print("警告: UI距离矩阵无效，创建空矩阵")
            self.uiMat = sp.csr_matrix((0, 0))

        if 'paper' in data.node_types:
            self.userNum = data['paper'].num_nodes
            print(f"用户(paper)数量: {self.userNum}")
        elif 'movie' in data.node_types:
            self.userNum = data['movie'].num_nodes
            print(f"用户(movie)数量: {self.userNum}")
        else:
            if hasattr(self.userMat, 'shape'):
                self.userNum = self.userMat.shape[0]
            else:
                self.userNum = 0

        if hasattr(self.itemMat, 'shape'):
            self.itemNum = self.itemMat.shape[0]
            print(f"物品数量: {self.itemNum}")
        else:
            self.itemNum = 0

        print(f"总节点数量: {self.userNum + self.itemNum}")

        if hasattr(self.userMat, 'shape') and self.userMat.shape != (self.userNum, self.userNum):
            print(f"警告: 用户矩阵形状 {self.userMat.shape} 与用户数量 {self.userNum} 不匹配")
            self.userMat = sp.csr_matrix((self.userNum, self.userNum))

        if hasattr(self.itemMat, 'shape') and self.itemMat.shape != (self.itemNum, self.itemNum):
            print(f"警告: 物品矩阵形状 {self.itemMat.shape} 与物品数量 {self.itemNum} 不匹配")
            self.itemMat = sp.csr_matrix((self.itemNum, self.itemNum))

        if hasattr(self.uiMat, 'shape') and self.uiMat.shape != (
                self.userNum + self.itemNum, self.userNum + self.itemNum):
            print(f"警告: UI矩阵形状 {self.uiMat.shape} 与总节点数量 {self.userNum + self.itemNum} 不匹配")
            self.uiMat = sp.csr_matrix((self.userNum + self.itemNum, self.userNum + self.itemNum))

        self.model = HGCL(
            self.args,
            self.userNum,
            self.itemNum,
            self.userMat, self.itemMat, self.uiMat,
            self.args.hide_dim,
            self.args.Layers)

    def get_embeddings(self):
        self.model.eval()

        with torch.no_grad():
            user_emb, item_emb, combined_user_emb, combined_item_emb = self.model()

        return {
            'user_embedding': user_emb,
            'item_embedding': item_emb,
            'combined_user_embedding': combined_user_emb,
            'combined_item_embedding': combined_item_emb
        }