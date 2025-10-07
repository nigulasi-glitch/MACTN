import os.path as osp
import numpy as np
import scipy.sparse as sp
import pickle
from tqdm import tqdm
from datasets import ACM, AMiner, FreeBase
import torch
from torch_geometric.data import HeteroData
from augment.semantic_augment import Hope
from args import make_args
from torch_geometric.data import (HeteroData, InMemoryDataset)
from typing import Callable, List, Optional
from scipy import io as sio
from torch_geometric.datasets import DBLP, IMDB


class MultiDataset(InMemoryDataset):
    def __init__(self, root: str, dataset_name: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, use_hgcl: bool = False):
        self.dataset_name = dataset_name
        self.use_hgcl = use_hgcl
        super().__init__(root, transform, pre_transform)

        if self.use_hgcl:
            hgcl_path = osp.join(self.processed_dir, f'data_with_hgcl_{dataset_name}.pt')
            if osp.exists(hgcl_path):
                try:
                    data = torch.load(hgcl_path)
                    self.data = data
                    print(f"已加载包含HGCL嵌入的{dataset_name}数据")
                except Exception as e:
                    print(f"加载HGCL数据时出错: {e}")
                    print("回退到原始数据")
                    self.data, self.slices = torch.load(self.processed_paths[0])
            else:
                print(f"未找到HGCL处理后的{dataset_name}数据，加载原始数据")
                self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        if self.dataset_name == 'acm':
            return ['ACM.mat']
        elif self.dataset_name == 'aminer':
            return ['labels.npy', 'pa.txt', 'pr.txt', 'features_0.npy', 'features_1.npy', 'features_2.npy']
        elif self.dataset_name == 'freebase':
            return ['labels.npy', 'ma.txt', 'md.txt', 'mw.txt', 'features_0.npy',
                    'features_1.npy', 'features_2.npy', 'features_3.npy']
        else:
            return []

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.dataset_name}.pt'

    def process(self):
        if self.use_hgcl:
            return

        data = HeteroData()

        if self.dataset_name == 'acm':
            raw_data = sio.loadmat(osp.join(self.raw_dir, 'ACM.mat'))
            p_vs_l = raw_data['PvsL']
            p_vs_a = raw_data['PvsA']
            p_vs_t = raw_data['PvsT']
            p_vs_p = raw_data['PvsP']
            p_vs_c = raw_data['PvsC']

            conf_ids = [0, 1, 9, 10, 13]
            label_ids = [0, 1, 2, 2, 1]

            p_vs_c_filter = p_vs_c[:, conf_ids]
            p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
            p_vs_c = p_vs_c[p_selected]
            p_vs_p = p_vs_p[p_selected].T[p_selected]
            a_selected = (p_vs_a[p_selected].sum(0) != 0).A1.nonzero()[0]
            p_vs_a = p_vs_a[p_selected].T[a_selected].T
            l_selected = (p_vs_l[p_selected].sum(0) != 0).A1.nonzero()[0]
            p_vs_l = p_vs_l[p_selected].T[l_selected].T
            t_selected = (p_vs_t[p_selected].sum(0) != 0).A1.nonzero()[0]
            p_vs_t = p_vs_t[p_selected].T[t_selected].T

            pc_p, pc_c = p_vs_c.nonzero()
            labels = np.zeros(len(p_selected), dtype=np.int64)
            for conf_id, label_id in zip(conf_ids, label_ids):
                labels[pc_p[pc_c == conf_id]] = label_id
            labels = torch.LongTensor(labels)

            data['paper'].x = torch.FloatTensor(p_vs_t.toarray())
            data['paper'].y = torch.LongTensor(labels)

            data['paper', 'author'].edge_index = torch.tensor(p_vs_a.nonzero(), dtype=torch.long)
            data['author', 'paper'].edge_index = torch.tensor(p_vs_a.transpose().nonzero(), dtype=torch.long)
            data['paper', 'subject'].edge_index = torch.tensor(p_vs_l.nonzero(), dtype=torch.long)
            data['subject', 'paper'].edge_index = torch.tensor(p_vs_l.transpose().nonzero(), dtype=torch.long)

        elif self.dataset_name == 'aminer':
            node_types = ['paper', 'author', 'reference']
            for i, node_type in enumerate(node_types):
                x = np.load(osp.join(self.raw_dir, f'features_{i}.npy'))
                data[node_type].x = torch.from_numpy(x).to(torch.float)

            labels = np.load(osp.join(self.raw_dir, 'labels.npy')).astype('int32')
            data['paper'].y = torch.from_numpy(labels)

            pa = np.loadtxt(osp.join(self.raw_dir, 'pa.txt'))
            pa = torch.from_numpy(pa).t()
            pr = np.loadtxt(osp.join(self.raw_dir, 'pr.txt'))
            pr = torch.from_numpy(pr).t()

            data['paper', 'reference'].edge_index = pr[[0, 1]].long()
            data['reference', 'paper'].edge_index = pr[[1, 0]].long()
            data['paper', 'author'].edge_index = pa[[0, 1]].long()
            data['author', 'paper'].edge_index = pa[[1, 0]].long()

        elif self.dataset_name == 'freebase':
            node_types = ['movie', 'actor', 'director', 'writer']
            for i, node_type in enumerate(node_types):
                x = np.load(osp.join(self.raw_dir, f'features_{i}.npy'))
                data[node_type].x = torch.from_numpy(x).to(torch.float)

            labels = np.load(osp.join(self.raw_dir, 'labels.npy')).astype('int32')
            data['movie'].y = torch.from_numpy(labels)

            ma = np.loadtxt(osp.join(self.raw_dir, 'ma.txt'))
            ma = torch.from_numpy(ma).t()
            md = np.loadtxt(osp.join(self.raw_dir, 'md.txt'))
            md = torch.from_numpy(md).t()
            mw = np.loadtxt(osp.join(self.raw_dir, 'mw.txt'))
            mw = torch.from_numpy(mw).t()

            data['movie', 'actor'].edge_index = ma[[0, 1]].long()
            data['actor', 'movie'].edge_index = ma[[1, 0]].long()
            data['movie', 'director'].edge_index = md[[0, 1]].long()
            data['director', 'movie'].edge_index = md[[1, 0]].long()
            data['movie', 'writer'].edge_index = mw[[0, 1]].long()
            data['writer', 'movie'].edge_index = mw[[1, 0]].long()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.dataset_name})'


class MetapathProcessor:
    def __init__(self, dataset_path, dataset_class, dataset_name):
        self.dataset_path = dataset_path
        self.dataset_class = dataset_class
        self.dataset_name = dataset_name
        self.dataset = dataset_class(dataset_path, dataset_name)
        self.data = self.dataset[0]
        self.processed_dir = osp.join(dataset_path, 'processed')

        if not osp.exists(self.processed_dir):
            import os
            os.makedirs(self.processed_dir)

        print(f"{dataset_name}数据节点类型:", self.data.node_types)
        print(f"{dataset_name}数据边类型:", self.data.edge_types)

    def extract_basic_relations(self):
        data = self.data

        node_counts = {}
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                node_counts[node_type] = data[node_type].x.shape[0]
            elif hasattr(data[node_type], 'num_nodes') and data[node_type].num_nodes is not None:
                node_counts[node_type] = data[node_type].num_nodes
            else:
                max_idx = 0
                for edge_type in data.edge_types:
                    if isinstance(edge_type, tuple) and len(edge_type) == 3:
                        src, rel, dst = edge_type
                        if src == node_type or dst == node_type:
                            edge_index = data[edge_type].edge_index.numpy()
                            if edge_index.size > 0:
                                if src == node_type:
                                    max_idx = max(max_idx, edge_index[0].max())
                                if dst == node_type:
                                    max_idx = max(max_idx, edge_index[1].max())
                node_counts[node_type] = max_idx + 1 if max_idx > 0 else 0

        print(f"{self.dataset_name}节点数量:", node_counts)

        relations = {}
        for edge_type in data.edge_types:
            if isinstance(edge_type, tuple) and len(edge_type) == 3:
                src, rel, dst = edge_type
                edge_index = data[edge_type].edge_index.numpy()

                if edge_index.size == 0:
                    relations[(src, dst)] = sp.csr_matrix((node_counts[src], node_counts[dst]))
                    relations[(dst, src)] = sp.csr_matrix((node_counts[dst], node_counts[src]))
                    continue

                mat = sp.dok_matrix((node_counts[src], node_counts[dst]))
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i]
                    dst_idx = edge_index[1, i]
                    mat[src_idx, dst_idx] = 1.0

                relations[(src, dst)] = mat.tocsr()
                relations[(dst, src)] = mat.T.tocsr()

        print(f"{self.dataset_name}提取的关系:", list(relations.keys()))
        return relations, node_counts

    def generate_metapath_relations(self, metapaths):
        relations, node_counts = self.extract_basic_relations()

        metapath_results = {}

        for i, metapath in enumerate(metapaths):
            print(f"\n处理元路径 {i + 1}: {metapath}")

            start_type = metapath[0][0]
            end_type = metapath[-1][1]

            metapath_mat = None
            for step in metapath:
                src, dst = step
                if (src, dst) not in relations:
                    print(f"警告: 关系 ({src, dst}) 不存在，跳过此元路径")
                    continue

                step_mat = relations[(src, dst)]

                if metapath_mat is None:
                    metapath_mat = step_mat
                else:
                    metapath_mat = metapath_mat.dot(step_mat)

            if metapath_mat is None:
                print(f"无法处理元路径 {metapath}，跳过")
                continue

            metapath_mat.data = np.ones_like(metapath_mat.data)

            intermediate_mats = {}
            for node_type in set([step[0] for step in metapath] + [step[1] for step in metapath]):
                if node_type != start_type and node_type != end_type:
                    pass

            distance_mat = {
                f'{start_type}_{start_type}': metapath_mat,
                f'{end_type}_{end_type}': metapath_mat.T,
            }

            for node_type, mat in intermediate_mats.items():
                distance_mat[f'{node_type}_{node_type}'] = mat

            for step in metapath:
                src, dst = step
                if (src, dst) in relations:
                    distance_mat[f'{src}_{dst}'] = relations[(src, dst)]

            metapath_name = ''.join([step[0][0].upper() for step in metapath]) + metapath[-1][1][0].upper()
            metapath_results[metapath_name] = {
                'distance_mat': distance_mat,
                'start_type': start_type,
                'end_type': end_type,
                'metapath': metapath
            }

            with open(osp.join(self.processed_dir, f'{self.dataset_name}_{metapath_name}_distanceMat.pkl'), 'wb') as f:
                pickle.dump(distance_mat, f)

            first_step = metapath[0]
            src, dst = first_step
            if (src, dst) in relations:
                ici_mat = relations[(src, dst)]
                with open(osp.join(self.processed_dir, f'{self.dataset_name}_{metapath_name}_ICI.pkl'), 'wb') as f:
                    pickle.dump(ici_mat, f)

            print(f"已保存{metapath_name}元路径的关系矩阵")

        return metapath_results

    def apply_hgcl_to_metapaths(self, metapath_results):
        args = make_args()
        hgcl_embeddings = {}

        for metapath_name, metapath_info in metapath_results.items():
            print(f"\n对{metapath_name}元路径应用HGCL模型...")

            distance_mat = metapath_info['distance_mat']
            start_type = metapath_info['start_type']
            end_type = metapath_info['end_type']

            user_type = start_type
            item_type = end_type

            user_num = distance_mat[f'{user_type}_{user_type}'].shape[0]
            item_num = distance_mat[f'{item_type}_{item_type}'].shape[0]

            user_mat = distance_mat[f'{user_type}_{user_type}']
            item_mat = distance_mat[f'{item_type}_{item_type}']

            ui_mat_key = f'{user_type}_{item_type}'
            if ui_mat_key in distance_mat:
                ui_mat = distance_mat[ui_mat_key]
            else:
                ui_mat = sp.csr_matrix((user_num, item_num))

            hope = Hope(args, self.data, distance_mat)

            embeddings = hope.get_embeddings()

            hgcl_embeddings[metapath_name] = {
                'user_embedding': embeddings['combined_user_embedding'],
                'item_embedding': embeddings['combined_item_embedding'],
                'user_type': user_type,
                'item_type': item_type
            }

            print(f"已完成{metapath_name}元路径的HGCL处理")

        return hgcl_embeddings

    def update_dataset_with_embeddings(self, hgcl_embeddings):
        updated_data = HeteroData()

        for node_type in self.data.node_types:
            if hasattr(self.data[node_type], 'x') and self.data[node_type].x is not None:
                updated_data[node_type].x = self.data[node_type].x.clone()
            if hasattr(self.data[node_type], 'y') and self.data[node_type].y is not None:
                updated_data[node_type].y = self.data[node_type].y.clone()
            for attr_name in self.data[node_type].keys():
                if attr_name not in ['x', 'y']:
                    updated_data[node_type][attr_name] = self.data[node_type][attr_name]

        for edge_type in self.data.edge_types:
            if hasattr(self.data[edge_type], 'edge_index') and self.data[edge_type].edge_index is not None:
                updated_data[edge_type].edge_index = self.data[edge_type].edge_index.clone()

        target_node = self.get_target_node_type()
        if target_node in updated_data.node_types and hgcl_embeddings:
            updated_data[target_node].hgcl_embeddings = {}
            for metapath_name, embedding_info in hgcl_embeddings.items():
                if 'user_embedding' in embedding_info:
                    updated_data[target_node].hgcl_embeddings[metapath_name] = embedding_info['user_embedding']
                elif 'combined_user_embedding' in embedding_info:
                    updated_data[target_node].hgcl_embeddings[metapath_name] = embedding_info['combined_user_embedding']

        torch.save(updated_data, osp.join(self.processed_dir, f'data_with_hgcl_{self.dataset_name}.pt'))

        return updated_data

    def get_target_node_type(self):
        if self.dataset_name == 'acm':
            return 'paper'
        elif self.dataset_name == 'aminer':
            return 'paper'
        elif self.dataset_name == 'freebase':
            return 'movie'
        elif self.dataset_name == 'dblp':
            return 'author'
        elif self.dataset_name == 'imdb':
            return 'movie'
        else:
            return 'paper'

    def process_metapaths(self, metapaths):
        metapath_results = self.generate_metapath_relations(metapaths)

        hgcl_embeddings = self.apply_hgcl_to_metapaths(metapath_results)

        updated_data = self.update_dataset_with_embeddings(hgcl_embeddings)

        return updated_data


class ACMMetapathProcessor(MetapathProcessor):
    def extract_basic_relations(self):
        data = self.data

        print("ACM数据边类型:", data.edge_types)

        node_counts = {}

        if hasattr(data['paper'], 'x') and data['paper'].x is not None:
            node_counts['paper'] = data['paper'].x.shape[0]
        else:
            node_counts['paper'] = data['paper'].num_nodes if hasattr(data['paper'], 'num_nodes') else 0

        max_author_idx = 0
        for edge_type in data.edge_types:
            if isinstance(edge_type, tuple) and len(edge_type) == 3:
                src, rel, dst = edge_type
                if dst == 'author' and src == 'paper':
                    edge_index = data[edge_type].edge_index.numpy()
                    if edge_index.size > 0:
                        max_author_idx = max(max_author_idx, edge_index[1].max())
        node_counts['author'] = max_author_idx + 1 if max_author_idx > 0 else 0

        max_subject_idx = 0
        for edge_type in data.edge_types:
            if isinstance(edge_type, tuple) and len(edge_type) == 3:
                src, rel, dst = edge_type
                if dst == 'subject' and src == 'paper':
                    edge_index = data[edge_type].edge_index.numpy()
                    if edge_index.size > 0:
                        max_subject_idx = max(max_subject_idx, edge_index[1].max())
        node_counts['subject'] = max_subject_idx + 1 if max_subject_idx > 0 else 0

        print("ACM节点数量:", node_counts)

        relations = {}

        for edge_type in data.edge_types:
            if isinstance(edge_type, tuple) and len(edge_type) == 3:
                src, rel, dst = edge_type
                edge_index = data[edge_type].edge_index.numpy()

                if edge_index.size == 0:
                    continue

                mat = sp.dok_matrix((node_counts[src], node_counts[dst]))
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i]
                    dst_idx = edge_index[1, i]
                    mat[src_idx, dst_idx] = 1.0

                relations[(src, dst)] = mat.tocsr()
                relations[(dst, src)] = mat.T.tocsr()

        print("ACM提取的关系:", list(relations.keys()))
        return relations, node_counts

    def generate_metapath_relations(self, metapaths):
        relations, node_counts = self.extract_basic_relations()

        metapath_results = {}

        for i, metapath in enumerate(metapaths):
            print(f"\n处理元路径 {i + 1}: {metapath}")

            if metapath == [('paper', 'author'), ('author', 'paper')]:
                metapath_name = 'PAP'
                start_type = 'paper'
                end_type = 'paper'

                if ('paper', 'author') not in relations or ('author', 'paper') not in relations:
                    print(f"警告: PAP元路径所需的关系不存在，跳过")
                    continue

                paper_author = relations[('paper', 'author')]
                author_paper = relations[('author', 'paper')]
                paper_paper = paper_author.dot(author_paper)
                paper_paper.data = np.ones_like(paper_paper.data)

                author_author = author_paper.dot(paper_author)
                author_author.data = np.ones_like(author_author.data)

                distance_mat = {
                    'paper_paper': paper_paper,
                    'author_author': author_author,
                    'paper_author': paper_author
                }

            elif metapath == [('paper', 'subject'), ('subject', 'paper')]:
                metapath_name = 'PSP'
                start_type = 'paper'
                end_type = 'paper'

                if ('paper', 'subject') not in relations or ('subject', 'paper') not in relations:
                    print(f"警告: PSP元路径所需的关系不存在，跳过")
                    continue

                paper_subject = relations[('paper', 'subject')]
                subject_paper = relations[('subject', 'paper')]
                paper_paper = paper_subject.dot(subject_paper)
                paper_paper.data = np.ones_like(paper_paper.data)

                subject_subject = subject_paper.dot(paper_subject)
                subject_subject.data = np.ones_like(subject_subject.data)

                distance_mat = {
                    'paper_paper': paper_paper,
                    'subject_subject': subject_subject,
                    'paper_subject': paper_subject
                }

            else:
                print(f"不支持的元路径: {metapath}")
                continue

            metapath_results[metapath_name] = {
                'distance_mat': distance_mat,
                'start_type': start_type,
                'end_type': end_type,
                'metapath': metapath
            }

            with open(osp.join(self.processed_dir, f'ACM_{metapath_name}_distanceMat.pkl'), 'wb') as f:
                pickle.dump(distance_mat, f)

            first_step = metapath[0]
            src, dst = first_step
            if (src, dst) in relations:
                ici_mat = relations[(src, dst)]
                with open(osp.join(self.processed_dir, f'ACM_{metapath_name}_ICI.pkl'), 'wb') as f:
                    pickle.dump(ici_mat, f)

            print(f"已保存{metapath_name}元路径的关系矩阵")

        return metapath_results

    def apply_hgcl_to_metapaths(self, metapath_results):
        args = make_args()
        hgcl_embeddings = {}

        for metapath_name, metapath_info in metapath_results.items():
            print(f"\n对{metapath_name}元路径应用HGCL模型...")

            distance_mat = metapath_info['distance_mat']
            start_type = metapath_info['start_type']
            end_type = metapath_info['end_type']

            user_type = start_type
            if metapath_name == 'PAP':
                item_type = 'author'
            elif metapath_name == 'PSP':
                item_type = 'subject'
            else:
                item_type = end_type

            user_num = distance_mat[f'{user_type}_{user_type}'].shape[0]
            item_num = distance_mat[f'{item_type}_{item_type}'].shape[0]

            user_mat = distance_mat[f'{user_type}_{user_type}']
            item_mat = distance_mat[f'{item_type}_{item_type}']
            ui_mat = distance_mat[f'{user_type}_{item_type}']

            hope = Hope(args, self.data, distance_mat)

            embeddings = hope.get_embeddings()

            if user_type == 'paper':
                hgcl_embeddings[metapath_name] = embeddings['combined_user_embedding']
            elif item_type == 'paper':
                hgcl_embeddings[metapath_name] = embeddings['combined_item_embedding']
            else:
                hgcl_embeddings[metapath_name] = embeddings['combined_user_embedding']

            print(f"已完成{metapath_name}元路径的HGCL处理")

        return hgcl_embeddings


class AMinerMetapathProcessor(MetapathProcessor):
    def generate_metapath_relations(self, metapaths):
        relations, node_counts = self.extract_basic_relations()

        metapath_results = {}

        for i, metapath in enumerate(metapaths):
            print(f"\n处理元路径 {i + 1}: {metapath}")

            if metapath == [('paper', 'reference'), ('reference', 'paper')]:
                metapath_name = 'PRP'
                start_type = 'paper'
                end_type = 'paper'

                if ('paper', 'reference') not in relations or ('reference', 'paper') not in relations:
                    print(f"警告: PRP元路径所需的关系不存在，跳过")
                    continue

                paper_reference = relations[('paper', 'reference')]
                reference_paper = relations[('reference', 'paper')]
                paper_paper = paper_reference.dot(reference_paper)
                paper_paper.data = np.ones_like(paper_paper.data)

                reference_reference = reference_paper.dot(paper_reference)
                reference_reference.data = np.ones_like(reference_reference.data)

                distance_mat = {
                    'paper_paper': paper_paper,
                    'reference_reference': reference_reference,
                    'paper_reference': paper_reference
                }

            elif metapath == [('paper', 'author'), ('author', 'paper')]:
                metapath_name = 'PAP'
                start_type = 'paper'
                end_type = 'paper'

                if ('paper', 'author') not in relations or ('author', 'paper') not in relations:
                    print(f"警告: PAP元路径所需的关系不存在，跳过")
                    continue

                paper_author = relations[('paper', 'author')]
                author_paper = relations[('author', 'paper')]
                paper_paper = paper_author.dot(author_paper)
                paper_paper.data = np.ones_like(paper_paper.data)

                author_author = author_paper.dot(paper_author)
                author_author.data = np.ones_like(author_author.data)

                distance_mat = {
                    'paper_paper': paper_paper,
                    'author_author': author_author,
                    'paper_author': paper_author
                }

            else:
                print(f"不支持的元路径: {metapath}")
                continue

            metapath_results[metapath_name] = {
                'distance_mat': distance_mat,
                'start_type': start_type,
                'end_type': end_type,
                'metapath': metapath
            }

            with open(osp.join(self.processed_dir, f'AMiner_{metapath_name}_distanceMat.pkl'), 'wb') as f:
                pickle.dump(distance_mat, f)

            first_step = metapath[0]
            src, dst = first_step
            if (src, dst) in relations:
                ici_mat = relations[(src, dst)]
                with open(osp.join(self.processed_dir, f'AMiner_{metapath_name}_ICI.pkl'), 'wb') as f:
                    pickle.dump(ici_mat, f)

            print(f"已保存{metapath_name}元路径的关系矩阵")

        return metapath_results


class FreeBaseMetapathProcessor(MetapathProcessor):
    def generate_metapath_relations(self, metapaths):
        relations, node_counts = self.extract_basic_relations()

        metapath_results = {}

        for i, metapath in enumerate(metapaths):
            print(f"\n处理元路径 {i + 1}: {metapath}")

            if metapath == [('movie', 'actor'), ('actor', 'movie')]:
                metapath_name = 'MAM'
                start_type = 'movie'
                end_type = 'movie'

                if ('movie', 'actor') not in relations or ('actor', 'movie') not in relations:
                    print(f"警告: MAM元路径所需的关系不存在，跳过")
                    continue

                movie_actor = relations[('movie', 'actor')]
                actor_movie = relations[('actor', 'movie')]
                movie_movie = movie_actor.dot(actor_movie)
                movie_movie.data = np.ones_like(movie_movie.data)

                actor_actor = actor_movie.dot(movie_actor)
                actor_actor.data = np.ones_like(actor_actor.data)

                distance_mat = {
                    'movie_movie': movie_movie,
                    'actor_actor': actor_actor,
                    'movie_actor': movie_actor
                }

            elif metapath == [('movie', 'director'), ('director', 'movie')]:
                metapath_name = 'MDM'
                start_type = 'movie'
                end_type = 'movie'

                if ('movie', 'director') not in relations or ('director', 'movie') not in relations:
                    print(f"警告: MDM元路径所需的关系不存在，跳过")
                    continue

                movie_director = relations[('movie', 'director')]
                director_movie = relations[('director', 'movie')]
                movie_movie = movie_director.dot(director_movie)
                movie_movie.data = np.ones_like(movie_movie.data)

                director_director = director_movie.dot(movie_director)
                director_director.data = np.ones_like(director_director.data)

                distance_mat = {
                    'movie_movie': movie_movie,
                    'director_director': director_director,
                    'movie_director': movie_director
                }

            elif metapath == [('movie', 'writer'), ('writer', 'movie')]:
                metapath_name = 'MWM'
                start_type = 'movie'
                end_type = 'movie'

                if ('movie', 'writer') not in relations or ('writer', 'movie') not in relations:
                    print(f"警告: MWM元路径所需的关系不存在，跳过")
                    continue

                movie_writer = relations[('movie', 'writer')]
                writer_movie = relations[('writer', 'movie')]
                movie_movie = movie_writer.dot(writer_movie)
                movie_movie.data = np.ones_like(movie_movie.data)

                writer_writer = writer_movie.dot(movie_writer)
                writer_writer.data = np.ones_like(writer_writer.data)

                distance_mat = {
                    'movie_movie': movie_movie,
                    'writer_writer': writer_writer,
                    'movie_writer': movie_writer
                }

            else:
                print(f"不支持的元路径: {metapath}")
                continue

            metapath_results[metapath_name] = {
                'distance_mat': distance_mat,
                'start_type': start_type,
                'end_type': end_type,
                'metapath': metapath
            }

            with open(osp.join(self.processed_dir, f'FreeBase_{metapath_name}_distanceMat.pkl'), 'wb') as f:
                pickle.dump(distance_mat, f)

            first_step = metapath[0]
            src, dst = first_step
            if (src, dst) in relations:
                ici_mat = relations[(src, dst)]
                with open(osp.join(self.processed_dir, f'FreeBase_{metapath_name}_ICI.pkl'), 'wb') as f:
                    pickle.dump(ici_mat, f)

            print(f"已保存{metapath_name}元路径的关系矩阵")

        return metapath_results

    def apply_hgcl_to_metapaths(self, metapath_results):
        args = make_args()
        hgcl_embeddings = {}

        for metapath_name, metapath_info in metapath_results.items():
            print(f"\n对{metapath_name}元路径应用HGCL模型...")

            distance_mat = metapath_info['distance_mat']

            hope = Hope(args, self.data, distance_mat)

            embeddings = hope.get_embeddings()

            if metapath_name == 'MAM':
                hgcl_embeddings[metapath_name] = embeddings['combined_user_embedding']
            elif metapath_name == 'MDM':
                hgcl_embeddings[metapath_name] = embeddings['combined_user_embedding']
            elif metapath_name == 'MWM':
                hgcl_embeddings[metapath_name] = embeddings['combined_user_embedding']
            else:
                hgcl_embeddings[metapath_name] = embeddings['combined_user_embedding']

            print(f"已完成{metapath_name}元路径的HGCL处理")

        return hgcl_embeddings


class DBLPMetapathProcessor(MetapathProcessor):
    def generate_metapath_relations(self, metapaths):
        relations, node_counts = self.extract_basic_relations()

        metapath_results = {}

        for i, metapath in enumerate(metapaths):
            print(f"\n处理元路径 {i + 1}: {metapath}")

            if metapath == [('author', 'paper'), ('paper', 'author')]:
                metapath_name = 'APA'
                start_type = 'author'
                end_type = 'author'

                if ('author', 'paper') not in relations or ('paper', 'author') not in relations:
                    print(f"警告: APA元路径所需的关系不存在，跳过")
                    continue

                author_paper = relations[('author', 'paper')]
                paper_author = relations[('paper', 'author')]
                author_author = author_paper.dot(paper_author)
                author_author.data = np.ones_like(author_author.data)

                paper_paper = paper_author.dot(author_paper)
                paper_paper.data = np.ones_like(paper_paper.data)

                distance_mat = {
                    'author_author': author_author,
                    'paper_paper': paper_paper,
                    'author_paper': author_paper
                }

            elif metapath == [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')]:
                metapath_name = 'APTPA'
                start_type = 'author'
                end_type = 'author'

                if ('author', 'paper') not in relations or ('paper', 'term') not in relations or \
                        ('term', 'paper') not in relations or ('paper', 'author') not in relations:
                    print(f"警告: APTPA元路径所需的关系不存在，跳过")
                    continue

                author_paper = relations[('author', 'paper')]
                paper_term = relations[('paper', 'term')]
                term_paper = relations[('term', 'paper')]
                paper_author = relations[('paper', 'author')]

                author_author = author_paper.dot(paper_term).dot(term_paper).dot(paper_author)
                author_author.data = np.ones_like(author_author.data)

                distance_mat = {
                    'author_author': author_author,
                    'author_paper': author_paper,
                    'paper_term': paper_term,
                    'term_paper': term_paper,
                    'paper_author': paper_author
                }

            elif metapath == [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'),
                              ('paper', 'author')]:
                metapath_name = 'APCPA'
                start_type = 'author'
                end_type = 'author'

                if ('author', 'paper') not in relations or ('paper', 'conference') not in relations or \
                        ('conference', 'paper') not in relations or ('paper', 'author') not in relations:
                    print(f"警告: APCPA元路径所需的关系不存在，跳过")
                    continue

                author_paper = relations[('author', 'paper')]
                paper_conference = relations[('paper', 'conference')]
                conference_paper = relations[('conference', 'paper')]
                paper_author = relations[('paper', 'author')]

                author_author = author_paper.dot(paper_conference).dot(conference_paper).dot(paper_author)
                author_author.data = np.ones_like(author_author.data)

                distance_mat = {
                    'author_author': author_author,
                    'author_paper': author_paper,
                    'paper_conference': paper_conference,
                    'conference_paper': conference_paper,
                    'paper_author': paper_author
                }

            else:
                print(f"不支持的元路径: {metapath}")
                continue

            metapath_results[metapath_name] = {
                'distance_mat': distance_mat,
                'start_type': start_type,
                'end_type': end_type,
                'metapath': metapath
            }

            with open(osp.join(self.processed_dir, f'DBLP_{metapath_name}_distanceMat.pkl'), 'wb') as f:
                pickle.dump(distance_mat, f)

            first_step = metapath[0]
            src, dst = first_step
            if (src, dst) in relations:
                ici_mat = relations[(src, dst)]
                with open(osp.join(self.processed_dir, f'DBLP_{metapath_name}_ICI.pkl'), 'wb') as f:
                    pickle.dump(ici_mat, f)

            print(f"已保存{metapath_name}元路径的关系矩阵")

        return metapath_results


class IMDBMetapathProcessor(MetapathProcessor):
    def generate_metapath_relations(self, metapaths):
        relations, node_counts = self.extract_basic_relations()

        metapath_results = {}

        for i, metapath in enumerate(metapaths):
            print(f"\n处理元路径 {i + 1}: {metapath}")

            if metapath == [('movie', 'director'), ('director', 'movie')]:
                metapath_name = 'MDM'
                start_type = 'movie'
                end_type = 'movie'

                if ('movie', 'director') not in relations or ('director', 'movie') not in relations:
                    print(f"警告: MDM元路径所需的关系不存在，跳过")
                    continue

                movie_director = relations[('movie', 'director')]
                director_movie = relations[('director', 'movie')]
                movie_movie = movie_director.dot(director_movie)
                movie_movie.data = np.ones_like(movie_movie.data)

                director_director = director_movie.dot(movie_director)
                director_director.data = np.ones_like(director_director.data)

                distance_mat = {
                    'movie_movie': movie_movie,
                    'director_director': director_director,
                    'movie_director': movie_director
                }

            elif metapath == [('movie', 'actor'), ('actor', 'movie')]:
                metapath_name = 'MAM'
                start_type = 'movie'
                end_type = 'movie'

                if ('movie', 'actor') not in relations or ('actor', 'movie') not in relations:
                    print(f"警告: MAM元路径所需的关系不存在，跳过")
                    continue

                movie_actor = relations[('movie', 'actor')]
                actor_movie = relations[('actor', 'movie')]
                movie_movie = movie_actor.dot(actor_movie)
                movie_movie.data = np.ones_like(movie_movie.data)

                actor_actor = actor_movie.dot(movie_actor)
                actor_actor.data = np.ones_like(actor_actor.data)

                distance_mat = {
                    'movie_movie': movie_movie,
                    'actor_actor': actor_actor,
                    'movie_actor': movie_actor
                }

            else:
                print(f"不支持的元路径: {metapath}")
                continue

            metapath_results[metapath_name] = {
                'distance_mat': distance_mat,
                'start_type': start_type,
                'end_type': end_type,
                'metapath': metapath
            }

            with open(osp.join(self.processed_dir, f'IMDB_{metapath_name}_distanceMat.pkl'), 'wb') as f:
                pickle.dump(distance_mat, f)

            first_step = metapath[0]
            src, dst = first_step
            if (src, dst) in relations:
                ici_mat = relations[(src, dst)]
                with open(osp.join(self.processed_dir, f'IMDB_{metapath_name}_ICI.pkl'), 'wb') as f:
                    pickle.dump(ici_mat, f)

            print(f"已保存{metapath_name}元路径的关系矩阵")

        return metapath_results


def get_dataset_semantic(dataset_name, use_hgcl=True):
    path = osp.join('D:/Heterogeneous graph learning/CTHNE-main/data', dataset_name)

    if dataset_name == 'dblp':
        dataset = MultiDataset(path, 'dblp', use_hgcl=True)
        metapaths = [
            [('author', 'paper'), ('paper', 'author')],
            [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],
            [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')]
        ]
        target = 'author'
    elif dataset_name == 'imdb':
        dataset = MultiDataset(path, 'imdb', use_hgcl=use_hgcl)
        metapaths = [
            [('movie', 'director'), ('director', 'movie')],
            [('movie', 'actor'), ('actor', 'movie')]
        ]
        target = 'movie'
    elif dataset_name == 'acm':
        dataset = MultiDataset(path, 'acm', use_hgcl=use_hgcl)
        metapaths = [
            [('paper', 'author'), ('author', 'paper')],
            [('paper', 'subject'), ('subject', 'paper')]
        ]
        target = 'paper'
    elif dataset_name == 'aminer':
        dataset = MultiDataset(path, 'aminer', use_hgcl=use_hgcl)
        metapaths = [
            [('paper', 'reference'), ('reference', 'paper')],
            [('paper', 'author'), ('author', 'paper')]
        ]
        target = 'paper'
    elif dataset_name == 'freebase':
        dataset = MultiDataset(path, 'freebase', use_hgcl=use_hgcl)
        metapaths = [
            [('movie', 'actor'), ('actor', 'movie')],
            [('movie', 'director'), ('director', 'movie')],
            [('movie', 'writer'), ('writer', 'movie')]
        ]
        target = 'movie'
    else:
        raise TypeError('不支持的数据库类型!')

    if use_hgcl:
        hgcl_path = osp.join(path, 'processed', f'data_with_hgcl_{dataset_name}.pt')
        if not osp.exists(hgcl_path):
            print(f"HGCL处理后的{dataset_name}数据不存在，开始处理元路径...")

            if dataset_name == 'acm':
                processor = ACMMetapathProcessor(path, MultiDataset, 'acm')
            elif dataset_name == 'aminer':
                processor = AMinerMetapathProcessor(path, MultiDataset, 'aminer')
            elif dataset_name == 'freebase':
                processor = FreeBaseMetapathProcessor(path, MultiDataset, 'freebase')
            elif dataset_name == 'dblp':
                processor = DBLPMetapathProcessor(path, MultiDataset, 'dblp')
            elif dataset_name == 'imdb':
                processor = IMDBMetapathProcessor(path, MultiDataset, 'imdb')
            else:
                print(f"不支持的数据库类型: {dataset_name}")
                return dataset, metapaths, target

            try:
                updated_data = processor.process_metapaths(metapaths)
                print(f"{dataset_name}元路径处理完成!")
            except Exception as e:
                print(f"处理过程中出错: {e}")
                import traceback
                traceback.print_exc()
                use_hgcl = False

        dataset = MultiDataset(path, dataset_name, use_hgcl=use_hgcl)

    return dataset, metapaths, target


if __name__ == "__main__":
    datasets = ['acm', 'aminer', 'freebase', 'dblp', 'imdb']

    for dataset_name in datasets:
        print(f"\n处理{dataset_name}数据集...")

        dataset, metapaths, target = get_dataset_semantic(dataset_name, use_hgcl=True)
        data = dataset[0]

        if hasattr(data[target], 'x') and data[target].x is not None:
            feat = data[target].x.numpy()
            print(f"{dataset_name}特征形状: {feat.shape}")

        if hasattr(data[target], 'hgcl_embeddings'):
            print(f"{dataset_name}的{target}节点的HGCL嵌入:")
            for metapath_name, embedding in data[target].hgcl_embeddings.items():
                print(f"  {metapath_name}: {embedding.shape}")