import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def mask_data(inputData):
    len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len for le in len_data]
    return us_msks, max_len

def complete_data(inputData):
    len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) for upois, le in zip(inputData, len_data)]
    us_pois.append([0] * max_len)
    return us_pois

def computeCoSeq(inputData, groups, dates_diff, max_len, si_dc, sample_order):
    length = len(inputData)
    inputData.append([0])
    out_edge_nodes_co, out_edge_weight_co, out_node_edges_co = [], [], []
    out_f_adj_nodes_seq, out_f_adj_weight_seq, out_b_adj_nodes_seq, out_b_adj_weight_seq = [],[],[],[]
    for i in range(length):
        items = np.flipud(np.unique(inputData[i]))
        con_sess_groups = [inputData[id] for id in groups[i]]
        cur_sess = inputData[i]
        con_dates_diff = dates_diff[i]
        edge_nodes_co, edge_weight_co, node_edges_co = generate_adjacency_co(cur_sess, con_sess_groups, con_dates_diff, items, si_dc, max_len)
        out_edge_nodes_co.append(edge_nodes_co)
        out_edge_weight_co.append(edge_weight_co)
        out_node_edges_co.append(node_edges_co)
        con_sess_groups.append(cur_sess)
        con_dates_diff = np.append(con_dates_diff, 0)
        f_adj_nodes_seq, f_adj_weight_seq, b_adj_nodes_seq, b_adj_weight_seq = generate_adjacency_seq(con_sess_groups, con_dates_diff, items, max_len, si_dc, sample_order)
        out_f_adj_nodes_seq.append(f_adj_nodes_seq)
        out_f_adj_weight_seq.append(f_adj_weight_seq)
        out_b_adj_nodes_seq.append(b_adj_nodes_seq)
        out_b_adj_weight_seq.append(b_adj_weight_seq)
    return out_edge_nodes_co, out_edge_weight_co, out_node_edges_co, out_f_adj_nodes_seq, out_f_adj_weight_seq, out_b_adj_nodes_seq, out_b_adj_weight_seq

class Data(Dataset):
    def __init__(self, data, groups_data, sample_order, si_dc):
        self.length = len(data[0])
        self.targets = np.asarray(data[1])
        mask, max_len = mask_data(data[0])
        self.mask = np.asarray(mask)
        re_inputs = complete_data(data[0])
        self.re_inputs = np.asarray(re_inputs)
        self.max_len = max_len
        self.groups = np.asarray(groups_data[0])
        self.neg_groups = np.asarray(groups_data[1])
        self.dates_diff = np.asarray(groups_data[2])
        edge_nodes_co, edge_weight_co, node_edges_co, f_adj_nodes_seq, f_adj_weight_seq, b_adj_nodes_seq, b_adj_weight_seq = computeCoSeq(data[0], self.groups, self.dates_diff, max_len, si_dc, sample_order)
        self.edge_nodes_co = edge_nodes_co
        self.edge_weight_co = edge_weight_co
        self.node_edges_co = node_edges_co
        self.f_adj_nodes_seq = f_adj_nodes_seq
        self.f_adj_weight_seq = f_adj_weight_seq
        self.b_adj_nodes_seq = b_adj_nodes_seq
        self.b_adj_weight_seq = b_adj_weight_seq

    def __getitem__(self, index):
        re_inputs = self.re_inputs[index]
        node = np.flipud(np.unique(re_inputs))
        items = list(node) + (self.max_len - len(node)) * [0]
        alias_re_inputs = [np.where(node == i)[0][0] for i in re_inputs]
        re_groups = self.re_inputs[self.groups[index]]
        re_neg_groups = self.re_inputs[self.neg_groups[index]]
        mask_groups = np.where(re_groups != 0, 1, 0)
        return alias_re_inputs, items, self.mask[index], self.targets[index], \
               re_inputs, re_groups, re_neg_groups, mask_groups, \
               self.edge_nodes_co[index], self.edge_weight_co[index], self.node_edges_co[index], \
               self.f_adj_nodes_seq[index], self.f_adj_weight_seq[index], self.b_adj_nodes_seq[index], self.b_adj_weight_seq[index]

    def __len__(self):
        return self.length

def collate_fn(batch):
    alias_re_inputs, items, mask, targets, re_inputs, re_groups, re_neg_groups, mask_groups, edge_nodes_co, edge_weight_co, node_edges_co, f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq = zip(*batch)
    edge_weight_co = pad_sequence(edge_weight_co, batch_first=True)
    max = edge_weight_co.shape[1]
    edge_nodes_co = [torch.cat((edge, torch.zeros(edge.shape[0], max-edge.shape[1])),1) for edge in edge_nodes_co]
    edge_nodes_co = pad_sequence(edge_nodes_co, batch_first=True).permute(0, 2, 1)
    node_edges_co = pad_sequence(node_edges_co, batch_first=True).permute(0, 2, 1)
    f_adjacency_nodes_seq = pad_sequence(f_adjacency_nodes_seq, batch_first=True).permute(0, 2, 3, 1)
    f_adjacency_weight_seq = pad_sequence(f_adjacency_weight_seq, batch_first=True).permute(0, 2, 3, 1)
    b_adjacency_nodes_seq = pad_sequence(b_adjacency_nodes_seq, batch_first=True).permute(0, 2, 3, 1)
    b_adjacency_weight_seq = pad_sequence(b_adjacency_weight_seq, batch_first=True).permute(0, 2, 3, 1)
    return torch.tensor(np.asarray(alias_re_inputs)), torch.tensor(items), torch.tensor(np.asarray(mask)), torch.tensor(targets), \
           torch.tensor(np.asarray(re_inputs)), torch.tensor(np.asarray(re_groups)), torch.tensor(np.asarray(re_neg_groups)), torch.tensor(np.asarray(mask_groups)), \
           edge_nodes_co, edge_weight_co, node_edges_co, f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq

def generate_adjacency_co(cur_sess, neig_re_group, neig_dates_diff, items, si_dc, max_len):
    node_edges_co_dict = {}
    for node in items:
        node_edges_co_dict[node] = []
    edge_nodes_co = []
    edge_nodes_co.append(torch.tensor([0]))
    id = 1
    for edge in neig_re_group:
        edge = np.unique(edge)
        edge_nodes_co.append(torch.tensor(edge))
        for node in edge:
            if node in items:
                node_edges_co_dict[node].append(id)
        id += 1
    m = id
    cur_edge_nodes_co = []
    le = len(cur_sess)
    if le <= 20:
        for i in np.arange(1, le + 1):
            for j in np.arange(le - i + 1):
                cur_edge_nodes_co.append(np.flipud(np.unique(cur_sess[j:j + i])))
    else:
        for i in np.arange(1,19):
            for j in np.arange(le-i+1):
                cur_edge_nodes_co.append(np.flipud(np.unique(cur_sess[j:j + i])))
        cur_edge_nodes_co.append(cur_sess)
    cur_edge_nodes_co = set(map(tuple, cur_edge_nodes_co))  # 去掉重复超边
    for edge in cur_edge_nodes_co:
        edge_nodes_co.append(torch.tensor(edge))
        for node in edge:
            node_edges_co_dict[node].append(id)
        id += 1
    edge_nodes_co = pad_sequence(edge_nodes_co, batch_first=True).permute(1, 0)

    edge_weight_co = [0]
    for ddates in neig_dates_diff:
        edge_weight_co.append(1/pow(si_dc, ddates))
    for _ in np.arange(id-m):
        edge_weight_co.append(1)
    edge_weight_co = torch.tensor(edge_weight_co)

    node_edges_co = []
    for value in node_edges_co_dict.values():
        node_edges_co.append(torch.tensor(value))
    for _ in np.arange(max_len - len(items)):
        node_edges_co.append(torch.tensor([0]))
    node_edges_co = pad_sequence(node_edges_co, batch_first=True).permute(1, 0)
    return edge_nodes_co, edge_weight_co, node_edges_co

def generate_adjacency_seq(temp_re_group, temp_dates_diff, items, max_len, si_dc, sample_order):
    f_temp_seq = {}
    b_temp_seq = {}
    for seq, ddates in zip(temp_re_group, temp_dates_diff):
        for i in np.arange(len(seq) - 1):
            item1 = seq[i]
            item2 = seq[i + 1]
            if item2 in f_temp_seq.keys():
                if item1 in f_temp_seq[item2][0]:
                    index = f_temp_seq[item2][0].index(item1)
                    f_temp_seq[item2][1][index] += 1 / pow(si_dc, ddates)
                else:
                    f_temp_seq[item2][0].append(item1)
                    f_temp_seq[item2][1].append(1 / pow(si_dc, ddates))
            else:
                f_temp_seq[item2] = [[item1], [1 / pow(si_dc, ddates)]]
            if item1 in b_temp_seq.keys():
                if item2 in b_temp_seq[item1][0]:
                    index = b_temp_seq[item1][0].index(item2)
                    b_temp_seq[item1][1][index] += 1 / pow(si_dc, ddates)
                else:
                    b_temp_seq[item1][0].append(item2)
                    b_temp_seq[item1][1].append(1 / pow(si_dc, ddates))
            else:
                b_temp_seq[item1] = [[item2], [1 / pow(si_dc, ddates)]]
    f_adjacency_nodes_seq, f_adjacency_weight_seq = compute_all_order_seq(f_temp_seq, items, sample_order, max_len)
    b_adjacency_nodes_seq, b_adjacency_weight_seq = compute_all_order_seq(b_temp_seq, items, sample_order, max_len)
    return f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq

def compute_all_order_seq(temp_seq, items, sample_order, max_len):
    all_adjacency_nodes_seq = []
    all_adjacency_weight_seq = []
    for item in items:
        adjacency_nodes_seq = []
        adjacency_weight_seq = []
        if item in temp_seq.keys():
            nodes = temp_seq[item][0]
            adjacency_nodes_seq.append(torch.tensor(nodes))
            adjacency_weight_seq.append(torch.tensor(temp_seq[item][1]))
            all_nodes = set(nodes)
            for _ in np.arange(1, sample_order):
                seq_k_n = []
                seq_k_w = []
                for n in nodes:
                    if n in temp_seq.keys():
                        for nn in temp_seq[n][0]:
                            ww = temp_seq[n][1][temp_seq[n][0].index(nn)]
                            if nn in seq_k_n:
                                seq_k_w[seq_k_n.index(nn)] += ww
                            else:
                                seq_k_n.append(nn)
                                seq_k_w.append(ww)
                temp_nodes = set(seq_k_n)
                nodes = temp_nodes - all_nodes
                if len(nodes) == 0:
                    adjacency_nodes_seq.append(torch.tensor([0]))
                    adjacency_weight_seq.append(torch.tensor([0]))
                    continue
                already_nodes = list(temp_nodes & all_nodes)
                for al_nodes in already_nodes:
                    index = seq_k_n.index(al_nodes)
                    del seq_k_n[index]
                    del seq_k_w[index]
                adjacency_nodes_seq.append(torch.tensor(seq_k_n))
                adjacency_weight_seq.append(torch.tensor(seq_k_w))
                all_nodes = temp_nodes | all_nodes
        else:
            for _ in np.arange(sample_order):
                adjacency_nodes_seq.append(torch.tensor([0]))
                adjacency_weight_seq.append(torch.tensor([0]))
        adjacency_nodes_seq = pad_sequence(adjacency_nodes_seq, batch_first=True).permute(1, 0)
        adjacency_weight_seq = pad_sequence(adjacency_weight_seq, batch_first=True).permute(1, 0)
        all_adjacency_nodes_seq.append(adjacency_nodes_seq)
        all_adjacency_weight_seq.append(adjacency_weight_seq)
    le = all_adjacency_nodes_seq[0].shape[0]
    for _ in np.arange(max_len-len(items)):
        all_adjacency_nodes_seq.append(torch.Tensor(le, sample_order).fill_(0))
        all_adjacency_weight_seq.append(torch.Tensor(le, sample_order).fill_(0))
    all_adjacency_nodes_seq = pad_sequence(all_adjacency_nodes_seq, batch_first=True).permute(1, 0, 2)
    all_adjacency_weight_seq = pad_sequence(all_adjacency_weight_seq, batch_first=True).permute(1, 0, 2)
    return all_adjacency_nodes_seq, all_adjacency_weight_seq
