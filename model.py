import math
import torch
from torch import nn
from torch.nn import Module

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MLEUP(Module):
    def __init__(self, opt, n_node):
        super(MLEUP, self).__init__()
        self.emb_size = opt.emb_size
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.f_embedding_pos = nn.Embedding(200, self.emb_size)
        self.b_embedding_pos = nn.Embedding(200, self.emb_size)
        self.sample_order = opt.sample_order
        self.dropout20 = nn.Dropout(0.2)

        self.W1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.q1 = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W2 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.q3 = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q4 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W3 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.W4 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.q5 = nn.Parameter(torch.Tensor(self.emb_size, 1))

        self.LN = nn.LayerNorm(self.emb_size)
        self.leakyrelu = nn.LeakyReLU(opt.alpha)

        self.k = opt.k
        self.item_model = MLP(self.emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, alias_re_inputs, items, mask, f_adjacency_nodes, f_adjacency_weight, b_adjacency_nodes, b_adjacency_weight):
        batch_size = mask.shape[0]
        max_len = mask.shape[1]
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)

        embedding = torch.cat([zeros, self.embedding.weight], 0)
        tar_embedding = embedding[items]
        # Item Representation Learning
        f_n_neighbor = f_adjacency_weight.shape[-1]
        f_adjacency_weight = f_adjacency_weight.reshape(batch_size, max_len, -1)
        f_adjacency_mask = torch.where(f_adjacency_weight != 0, 1, 0)
        f_embedding_pos = self.f_embedding_pos.weight[:self.sample_order].reshape(1, 1, self.sample_order, 1, self.emb_size).repeat(batch_size, max_len, 1, f_n_neighbor, 1)
        neig_f_embedding = self.W1(torch.cat([f_embedding_pos, embedding[f_adjacency_nodes]], -1)).reshape(batch_size, max_len, -1, self.emb_size)
        neig_f_embedding = torch.tanh(neig_f_embedding)
        neig_f_embedding = neig_f_embedding * f_adjacency_mask.unsqueeze(-1)
        b_n_neighbor = b_adjacency_weight.shape[-1]
        b_adjacency_weight = b_adjacency_weight.reshape(batch_size, max_len, -1)
        b_adjacency_mask = torch.where(b_adjacency_weight != 0, 1, 0)
        b_embedding_pos = self.b_embedding_pos.weight[:self.sample_order].reshape(1, 1, self.sample_order, 1, self.emb_size).repeat(batch_size, max_len, 1, b_n_neighbor, 1)
        neig_b_embedding = self.W2(torch.cat([b_embedding_pos, embedding[b_adjacency_nodes]], -1)).reshape(batch_size, max_len, -1, self.emb_size)
        neig_b_embedding = torch.tanh(neig_b_embedding)
        neig_b_embedding = neig_b_embedding * b_adjacency_mask.unsqueeze(-1)

        neig_f_embedding = self.weighted_GAT(tar_embedding, neig_f_embedding, f_adjacency_weight, self.q1, self.q2)
        neig_b_embedding = self.weighted_GAT(tar_embedding, neig_b_embedding, b_adjacency_weight, self.q3, self.q4)
        neig_embedding = neig_f_embedding + neig_b_embedding

        gate = torch.sigmoid(self.W3(torch.cat([neig_embedding, tar_embedding], -1)))
        final_embedding = gate * neig_embedding + (1 - gate) * tar_embedding
        final_embedding = self.dropout20(final_embedding)

        # Session Representation Learning
        embedding_pos = self.f_embedding_pos.weight[:max_len].unsqueeze(0).repeat(batch_size, 1, 1)
        get = lambda index: final_embedding[index][alias_re_inputs[index]]
        alias_final_embedding = torch.stack([get(i) for i in torch.arange(batch_size)])
        final_embedding_pos = torch.tanh(self.W4(torch.cat([alias_final_embedding, embedding_pos], -1)))
        final_embedding_pos = final_embedding_pos * mask.unsqueeze(-1)
        attention_mask = (1.0 - mask) * -10000.0
        alpha5 = torch.matmul(self.leakyrelu(final_embedding_pos), self.q5).squeeze(-1)
        alpha5 = alpha5 + attention_mask
        alpha5 = torch.softmax(alpha5, -1).unsqueeze(-1)
        sess_embedding = torch.sum(alpha5 * final_embedding_pos, 1)
        sess_embedding = self.dropout20(self.LN(sess_embedding))

        # Position Encoding Enhanced Recommendation
        scores_p = torch.mm(sess_embedding, torch.transpose(embedding, 1, 0))

        # De-popularity Bias Recommendation
        y_i = embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        y_i = self.item_model(y_i).squeeze(-1)
        scores_d = scores_p * torch.sigmoid(y_i) - self.k * torch.sigmoid(y_i)
        return scores_p, scores_d

    def weighted_GAT(self, tar_embedding, neig_embedding, adjacency_weight, q1, q2):
        attention_adjacency_mask = torch.where(adjacency_weight != 0, 0, -10000)
        alpha = torch.matmul(torch.cat([tar_embedding.unsqueeze(2) * neig_embedding, adjacency_weight.unsqueeze(-1)], -1), q1)
        alpha = torch.matmul(self.leakyrelu(alpha), q2).squeeze(-1)
        alpha = alpha + attention_adjacency_mask
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        neig_embedding_i = torch.sum(alpha * neig_embedding, 2)
        return neig_embedding_i

def forward(model, data):
    alias_re_inputs, items, mask, targets, f_adjacency_nodes, f_adjacency_weight, b_adjacency_nodes, b_adjacency_weight = data
    alias_re_inputs = trans_to_cuda(alias_re_inputs).long()
    items = trans_to_cuda(items).long()
    mask = trans_to_cuda(mask).long()
    f_adjacency_nodes = trans_to_cuda(f_adjacency_nodes).long()
    f_adjacency_weight = trans_to_cuda(f_adjacency_weight).float()
    b_adjacency_nodes = trans_to_cuda(b_adjacency_nodes).long()
    b_adjacency_weight = trans_to_cuda(b_adjacency_weight).float()
    targets = trans_to_cuda(targets).long()
    scores_p, scores_d = model(alias_re_inputs, items, mask, f_adjacency_nodes, f_adjacency_weight, b_adjacency_nodes, b_adjacency_weight)
    return targets, scores_p, scores_d