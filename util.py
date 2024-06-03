import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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

def computeAdj(inputData, groups, dates_diff, max_len, si_dc, sample_order):
    length = len(inputData)
    inputData.append([0])
    out_f_adj_nodes, out_f_adj_weight, out_b_adj_nodes, out_b_adj_weight = [],[],[],[]
    for i in tqdm(range(length)):
        items = np.flipud(np.unique(inputData[i]))
        con_sess_groups = [inputData[id] for id in groups[i]]
        cur_sess = inputData[i]
        con_dates_diff = dates_diff[i]
        con_sess_groups.append(cur_sess)
        con_dates_diff = np.append(con_dates_diff, 0)
        f_adj_nodes, f_adj_weight, b_adj_nodes, b_adj_weight = generate_adjacency(con_sess_groups, con_dates_diff, items, max_len, si_dc, sample_order)
        out_f_adj_nodes.append(f_adj_nodes)
        out_f_adj_weight.append(f_adj_weight)
        out_b_adj_nodes.append(b_adj_nodes)
        out_b_adj_weight.append(b_adj_weight)
    return out_f_adj_nodes, out_f_adj_weight, out_b_adj_nodes, out_b_adj_weight

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
        self.dates_diff = np.asarray(groups_data[2])
        f_adj_nodes, f_adj_weight, b_adj_nodes, b_adj_weight = computeAdj(data[0], self.groups, self.dates_diff, max_len, si_dc, sample_order)
        self.f_adj_nodes = f_adj_nodes
        self.f_adj_weight = f_adj_weight
        self.b_adj_nodes = b_adj_nodes
        self.b_adj_weight = b_adj_weight

    def __getitem__(self, index):
        re_inputs = self.re_inputs[index]
        node = np.flipud(np.unique(re_inputs))
        items = list(node) + (self.max_len - len(node)) * [0]
        alias_re_inputs = [np.where(node == i)[0][0] for i in re_inputs]
        return alias_re_inputs, items, self.mask[index], self.targets[index], \
               self.f_adj_nodes[index], self.f_adj_weight[index], self.b_adj_nodes[index], self.b_adj_weight[index]

    def __len__(self):
        return self.length

def collate_fn(batch):
    alias_re_inputs, items, mask, targets, f_adjacency_nodes, f_adjacency_weight, b_adjacency_nodes, b_adjacency_weight = zip(*batch)
    f_adjacency_nodes = pad_sequence(f_adjacency_nodes, batch_first=True).permute(0, 2, 3, 1)
    f_adjacency_weight = pad_sequence(f_adjacency_weight, batch_first=True).permute(0, 2, 3, 1)
    b_adjacency_nodes = pad_sequence(b_adjacency_nodes, batch_first=True).permute(0, 2, 3, 1)
    b_adjacency_weight = pad_sequence(b_adjacency_weight, batch_first=True).permute(0, 2, 3, 1)
    return torch.tensor(np.asarray(alias_re_inputs)), torch.tensor(items), torch.tensor(np.asarray(mask)), torch.tensor(targets), \
           f_adjacency_nodes, f_adjacency_weight, b_adjacency_nodes, b_adjacency_weight

def generate_adjacency(temp_re_group, temp_dates_diff, items, max_len, si_dc, sample_order):
    f_temp = {}
    b_temp = {}
    for seq, ddates in zip(temp_re_group, temp_dates_diff):
        for i in np.arange(len(seq) - 1):
            item1 = seq[i]
            item2 = seq[i + 1]
            if item2 in f_temp.keys():
                if item1 in f_temp[item2][0]:
                    index = f_temp[item2][0].index(item1)
                    f_temp[item2][1][index] += 1 / pow(si_dc, ddates)
                else:
                    f_temp[item2][0].append(item1)
                    f_temp[item2][1].append(1 / pow(si_dc, ddates))
            else:
                f_temp[item2] = [[item1], [1 / pow(si_dc, ddates)]]
            if item1 in b_temp.keys():
                if item2 in b_temp[item1][0]:
                    index = b_temp[item1][0].index(item2)
                    b_temp[item1][1][index] += 1 / pow(si_dc, ddates)
                else:
                    b_temp[item1][0].append(item2)
                    b_temp[item1][1].append(1 / pow(si_dc, ddates))
            else:
                b_temp[item1] = [[item2], [1 / pow(si_dc, ddates)]]
    f_adjacency_nodes, f_adjacency_weight = compute_all_order(f_temp, items, sample_order, max_len)
    b_adjacency_nodes, b_adjacency_weight = compute_all_order(b_temp, items, sample_order, max_len)
    return f_adjacency_nodes, f_adjacency_weight, b_adjacency_nodes, b_adjacency_weight

def compute_all_order(temp, items, sample_order, max_len):
    all_adjacency_nodes = []
    all_adjacency_weight = []
    for item in items:
        adjacency_nodes = []
        adjacency_weight = []
        if item in temp.keys():
            nodes = temp[item][0]
            adjacency_nodes.append(torch.tensor(nodes))
            adjacency_weight.append(torch.tensor(temp[item][1]))
            all_nodes = set(nodes)
            for _ in np.arange(1, sample_order):
                seq_k_n = []
                seq_k_w = []
                for n in nodes:
                    if n in temp.keys():
                        for nn in temp[n][0]:
                            ww = temp[n][1][temp[n][0].index(nn)]
                            if nn in seq_k_n:
                                seq_k_w[seq_k_n.index(nn)] += ww
                            else:
                                seq_k_n.append(nn)
                                seq_k_w.append(ww)
                temp_nodes = set(seq_k_n)
                nodes = temp_nodes - all_nodes
                if len(nodes) == 0:
                    adjacency_nodes.append(torch.tensor([0]))
                    adjacency_weight.append(torch.tensor([0]))
                    continue
                already_nodes = list(temp_nodes & all_nodes)
                for al_nodes in already_nodes:
                    index = seq_k_n.index(al_nodes)
                    del seq_k_n[index]
                    del seq_k_w[index]
                adjacency_nodes.append(torch.tensor(seq_k_n))
                adjacency_weight.append(torch.tensor(seq_k_w))
                all_nodes = temp_nodes | all_nodes
        else:
            for _ in np.arange(sample_order):
                adjacency_nodes.append(torch.tensor([0]))
                adjacency_weight.append(torch.tensor([0]))
        adjacency_nodes = pad_sequence(adjacency_nodes, batch_first=True).permute(1, 0)
        adjacency_weight = pad_sequence(adjacency_weight, batch_first=True).permute(1, 0)
        all_adjacency_nodes.append(adjacency_nodes)
        all_adjacency_weight.append(adjacency_weight)
    le = all_adjacency_nodes[0].shape[0]
    for _ in np.arange(max_len-len(items)):
        all_adjacency_nodes.append(torch.Tensor(le, sample_order).fill_(0))
        all_adjacency_weight.append(torch.Tensor(le, sample_order).fill_(0))
    all_adjacency_nodes = pad_sequence(all_adjacency_nodes, batch_first=True).permute(1, 0, 2)
    all_adjacency_weight = pad_sequence(all_adjacency_weight, batch_first=True).permute(1, 0, 2)
    return all_adjacency_nodes, all_adjacency_weight
