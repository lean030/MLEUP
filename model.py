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


class FL(Module):
    def __init__(self, num_f, emb_size, n_node):
        super(FL, self).__init__()
        self.num_f = num_f
        self.n_node = n_node
        self.emb_size = emb_size
        self.embedding_f = nn.Embedding(self.num_f, self.emb_size)
        self.u = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W = nn.Linear(self.emb_size * 2, self.emb_size)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout70 = nn.Dropout(0.7)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, adjacency_fi, embedding_i):
        neigh_embedding = embedding_i[adjacency_fi]
        agg_embedding = self.agg_feature(adjacency_fi, neigh_embedding)
        new_embedding_f = self.upd_feature(self.embedding_f.weight, agg_embedding)
        return new_embedding_f

    def agg_feature(self, adjacency_fi, neigh_embedding):
        attention_mask = torch.where(adjacency_fi != 0, 0, -10000)
        alpha = torch.matmul(neigh_embedding, self.u).squeeze(-1)
        alpha = alpha + attention_mask
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        neigh_embedding = torch.sum(alpha * neigh_embedding, 1)
        return self.dropout70(neigh_embedding)

    def upd_feature(self, emb1, emb2):
        gate = torch.sigmoid(self.W(torch.cat([emb1, emb2], 1)))
        new_embedding = gate * emb1 + (1 - gate) * emb2
        return self.dropout50(new_embedding)

class MLEUP(Module):
    def __init__(self, opt, n_node, all_feature, adjacency_fi, adjacency_if):
        super(MLEUP, self).__init__()
        self.m = opt.m
        self.emb_size = opt.emb_size
        self.batch_size = opt.batch_size
        self.n_node = n_node
        self.embedding_i = nn.Embedding(self.n_node, self.emb_size)
        self.f_embedding_pos = nn.Embedding(200, self.emb_size)
        self.b_embedding_pos = nn.Embedding(200, self.emb_size)

        self.sample_order = opt.sample_order
        self.adjacency_fi = adjacency_fi
        self.adjacency_if = adjacency_if
        self.all_feature = all_feature
        self.n_feature = len(all_feature)
        self.all_feature_model = []
        for i, num in zip(range(self.n_feature), self.all_feature):
            fm = FL(num, self.emb_size, self.n_node)
            self.add_module('feature_model_{}'.format(i), fm)
            self.all_feature_model.append(fm)
        self.W1_feature = nn.Linear(self.emb_size * self.n_feature, self.emb_size)
        self.query = nn.Linear(self.emb_size , self.emb_size)
        self.key = nn.Linear(self.emb_size , self.emb_size)
        self.value = nn.Linear(self.emb_size , self.emb_size)
        self.W2_feature = nn.Linear(self.emb_size * 2, self.emb_size)
        self.num_heads = opt.num_heads  # 4
        self.attention_head_size = int(self.emb_size / self.num_heads)
        self.dropout20 = nn.Dropout(0.2)
        self.u_feature = nn.Parameter(torch.Tensor(self.emb_size,1))
        self.beta = opt.beta

        self.q1_co = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.q2_co = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q3_co = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W4_co = nn.Linear(self.emb_size * 2, self.emb_size)
        self.W5_co = nn.Linear(self.emb_size * 2, self.emb_size)
        self.W6_co = nn.Linear(self.emb_size, self.emb_size)
        self.W7_co = nn.Linear(self.emb_size, self.emb_size)
        self.W8_co = nn.Linear(self.emb_size, self.emb_size)
        self.q4_co = nn.Parameter(torch.Tensor(self.emb_size, 1))

        self.W1_seq = nn.Linear(self.emb_size * 2, self.emb_size)
        self.q1_seq = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q2_seq = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W2_seq = nn.Linear(self.emb_size * 2, self.emb_size)
        self.q3_seq = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q4_seq = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W3_seq = nn.Linear(self.emb_size * 2, self.emb_size)
        self.W4_seq = nn.Linear(self.emb_size * 2, self.emb_size)
        self.W5_seq = nn.Linear(self.emb_size, self.emb_size)
        self.W6_seq = nn.Linear(self.emb_size, self.emb_size)
        self.W7_seq = nn.Linear(self.emb_size, self.emb_size)
        self.q5_seq = nn.Parameter(torch.Tensor(self.emb_size, 1))

        self.W1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.LN = nn.LayerNorm(self.emb_size)
        self.W2 = nn.Linear(self.emb_size, self.emb_size)
        self.W3 = nn.Linear(self.emb_size, self.emb_size)
        self.W4 = nn.Linear(self.emb_size, self.emb_size)
        self.W5 = nn.Linear(self.emb_size, self.emb_size)
        self.W6 = nn.Linear(self.emb_size, self.emb_size)
        self.W7 = nn.Linear(self.emb_size, self.emb_size)
        self.W8 = nn.Linear(self.emb_size, self.emb_size)
        self.W9 = nn.Linear(self.emb_size, self.emb_size)
        self.W10 = nn.Linear(self.emb_size, self.emb_size)
        self.W11 = nn.Linear(self.emb_size, self.emb_size)
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, alias_re_inputs, items, mask, re_inputs, re_groups, re_neg_groups, mask_groups, edge_nodes_co, edge_weight_co, node_edges_co, f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq):
        batch_size = mask.shape[0]
        max_len = mask.shape[1]
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        sample_order = self.sample_order

        embedding_i = torch.cat([zeros, self.embedding_i.weight], 0)
        # Attribute embedding learning
        # (1) Modeling global attribute-item relationship   (2) Concatenating multiple attribute values of an item
        con_embedding_f = torch.cuda.FloatTensor([])
        for i in range(self.n_feature):
            new_embedding_f = self.all_feature_model[i](trans_to_cuda(torch.tensor(self.adjacency_fi[i])).long(), embedding_i)
            con_embedding_f = torch.cat([con_embedding_f, new_embedding_f[trans_to_cuda(torch.tensor(self.adjacency_if[i])).long()]], -1)
        embedding_f = self.W1_feature(con_embedding_f)
        embedding_f = torch.cat([zeros, embedding_f], 0)
        # (3) Modeling attribute-attribute relationships within a session
        final_embedding_f = self.generate_embedding_f(embedding_f, re_inputs, mask, max_len, batch_size)
        # Stable user preference extraction
        sess_embedding_f = self.generate_sess_embedding_f(final_embedding_f, mask)
        # Neighbor-based contrastive self-supervised attribute learning task
        s_loss = 0
        one = torch.cuda.FloatTensor(batch_size).fill_(1)
        for i in torch.arange(self.m):
            mask_g = mask_groups[:,i,:].squeeze(1)
            pos_sess_embedding_f = self.generate_sess_embedding_f(self.generate_embedding_f(embedding_f, re_groups[:,i,:].squeeze(1), mask_g, max_len, batch_size), mask_g)
            pos = self.discrim_feature(sess_embedding_f, pos_sess_embedding_f)
            neg_sess_embedding_f = self.generate_sess_embedding_f(self.generate_embedding_f(embedding_f, re_neg_groups[:,i,:].squeeze(1), mask_g, max_len, batch_size), mask_g)
            neg = self.discrim_feature(sess_embedding_f, neg_sess_embedding_f)
            s_loss = s_loss + torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg))))

        # Item embedding learning
        tar_embedding_i = embedding_i[items]
        # Item co-occurrence relationship encoder
        # (1) Node-to-hyperedge propagation
        edge_embedding = embedding_i[edge_nodes_co]
        edge_nodes_mask_co = torch.where(edge_nodes_co != 0, 1, 0)
        edge_nodes_att_mask_co = (1.0 - edge_nodes_mask_co) * -10000.0
        num = torch.sum(edge_nodes_mask_co, 2)
        edge_embedding_query = torch.sum(edge_embedding.float(), 2) / torch.where(num == 0, float('inf'), num).unsqueeze(2)
        alpha = torch.matmul(edge_embedding * edge_embedding_query.unsqueeze(2), self.q1_co).squeeze(-1)
        alpha = alpha + edge_nodes_att_mask_co
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        edge_embedding = torch.sum(alpha * edge_embedding, 2)
        # (2) Hyperedge-to-node propagation
        node_embedding = torch.cat([edge_embedding[i][node_edges_co[i]].unsqueeze(0) for i in torch.arange(batch_size)])
        node_edges_weight_co = torch.cat([edge_weight_co[i][node_edges_co[i]].unsqueeze(0) for i in torch.arange(batch_size)])
        final_embedding_i_co = self.weighted_GAT(tar_embedding_i, node_embedding, node_edges_weight_co, self.q2_co, self.q3_co)
        # Item sequential relationship encoder
        # (1) Concatenating relative position embeddings
        f_n_neighbor = f_adjacency_weight_seq.shape[-1]
        f_adjacency_weight_seq = f_adjacency_weight_seq.reshape(batch_size, max_len, -1)
        f_adjacency_seq_mask = torch.where(f_adjacency_weight_seq != 0, 1, 0)
        f_embedding_pos_seq = self.f_embedding_pos.weight[:sample_order].reshape(1, 1, sample_order, 1, self.emb_size).repeat(batch_size, max_len, 1, f_n_neighbor, 1)
        neig_f_embedding_i_seq = self.W1_seq(torch.cat([f_embedding_pos_seq, embedding_i[f_adjacency_nodes_seq]], -1)).reshape(batch_size, max_len, -1, self.emb_size)
        neig_f_embedding_i_seq = torch.tanh(neig_f_embedding_i_seq)
        neig_f_embedding_i_seq = neig_f_embedding_i_seq * f_adjacency_seq_mask.unsqueeze(-1)
        b_n_neighbor = b_adjacency_weight_seq.shape[-1]
        b_adjacency_weight_seq = b_adjacency_weight_seq.reshape(batch_size, max_len, -1)
        b_adjacency_seq_mask = torch.where(b_adjacency_weight_seq != 0, 1, 0)
        b_embedding_pos_seq = self.b_embedding_pos.weight[:sample_order].reshape(1, 1, sample_order, 1, self.emb_size).repeat(batch_size, max_len, 1, b_n_neighbor, 1)
        neig_b_embedding_i_seq = self.W2_seq(torch.cat([b_embedding_pos_seq, embedding_i[b_adjacency_nodes_seq]], -1)).reshape(batch_size, max_len, -1, self.emb_size)
        neig_b_embedding_i_seq = torch.tanh(neig_b_embedding_i_seq)
        neig_b_embedding_i_seq = neig_b_embedding_i_seq * b_adjacency_seq_mask.unsqueeze(-1)
        # (2) learning the information propagated from high-order adjacent edges
        neig_f_embedding_i_seq = self.weighted_GAT(tar_embedding_i, neig_f_embedding_i_seq, f_adjacency_weight_seq, self.q1_seq, self.q2_seq)
        neig_b_embedding_i_seq = self.weighted_GAT(tar_embedding_i, neig_b_embedding_i_seq, b_adjacency_weight_seq, self.q3_seq, self.q4_seq)
        neig_embedding_i_seq = neig_f_embedding_i_seq + neig_b_embedding_i_seq
        # (3) Updating node embedding
        gate3 = torch.sigmoid(self.W3_seq(torch.cat([neig_embedding_i_seq, tar_embedding_i], -1)))
        final_embedding_i_seq = gate3 * neig_embedding_i_seq + (1 - gate3) * tar_embedding_i
        final_embedding_i_seq = self.dropout20(final_embedding_i_seq)

        # Dynamic user preference extraction
        # (2) Concatenating forward position embeddings
        embedding_pos_seq = self.f_embedding_pos.weight[:max_len].unsqueeze(0).repeat(batch_size, 1, 1)
        get = lambda index: final_embedding_i_co[index][alias_re_inputs[index]]
        embedding_i_co = torch.stack([get(i) for i in torch.arange(batch_size)])
        embedding_i_co = torch.tanh(self.W5_co(torch.cat([embedding_i_co, embedding_pos_seq], -1)))
        embedding_i_co = embedding_i_co * mask.unsqueeze(-1)
        get = lambda index: final_embedding_i_seq[index][alias_re_inputs[index]]
        embedding_i_seq = torch.stack([get(i) for i in torch.arange(batch_size)])
        embedding_i_seq = torch.tanh(self.W4_seq(torch.cat([embedding_i_seq, embedding_pos_seq], -1)))
        embedding_i_seq = embedding_i_seq * mask.unsqueeze(-1)
        # (2) Learning session embeddings
        attention_mask = (1.0 - mask) * -10000.0
        f3 = self.W6_co(embedding_i_co)
        f4 = self.W7_co(final_embedding_f)
        f5 = self.W8_co(sess_embedding_f).unsqueeze(1)
        alpha2 = torch.matmul(self.leakyrelu(f3 + f4 + f5), self.q4_co).squeeze(-1)
        alpha2 = alpha2 + attention_mask
        alpha2 = torch.softmax(alpha2, -1).unsqueeze(-1)
        sess_embedding_i_co = torch.sum(alpha2 * embedding_i_co, 1)
        f6 = self.W5_seq(embedding_i_seq)
        f7 = self.W6_seq(final_embedding_f)
        f8 = self.W7_seq(sess_embedding_f).unsqueeze(1)
        alpha5 = torch.matmul(self.leakyrelu(f6 + f7 + f8), self.q5_seq).squeeze(-1)
        alpha5 = alpha5 + attention_mask
        alpha5 = torch.softmax(alpha5, -1).unsqueeze(-1)
        sess_embedding_i_seq = torch.sum(alpha5 * embedding_i_seq, 1)
        # (3) Integrating the two
        gate4 = torch.sigmoid(self.W1(torch.cat([sess_embedding_i_co, sess_embedding_i_seq], -1)))
        sess_embedding_i = gate4 * sess_embedding_i_co + (1 - gate4) * sess_embedding_i_seq
        sess_embedding_i = self.dropout20(self.LN(sess_embedding_i))

        # User preferences fusion
        sess_embedding_f = self.dropout20(self.LN(sess_embedding_f))
        m_c = torch.tanh(self.W2(sess_embedding_f * sess_embedding_i))
        m_j = torch.tanh(self.W3(sess_embedding_f + sess_embedding_i))
        r_i = torch.sigmoid(self.W4(m_c) + self.W5(m_j))
        r_f = torch.sigmoid(self.W6(m_c) + self.W7(m_j))
        m_f = torch.tanh(self.W8(sess_embedding_f * r_f) + self.W9((1 - r_i) * sess_embedding_i))
        m_i = torch.tanh(self.W10(sess_embedding_i * r_i) + self.W11((1 - r_f) * sess_embedding_f))
        f_pre = (sess_embedding_f + m_i) * m_f
        i_pre = (sess_embedding_i + m_f) * m_i
        scores_f = torch.mm(f_pre, torch.transpose(embedding_f, 1, 0))
        scores_i = torch.mm(i_pre, torch.transpose(embedding_i, 1, 0))
        scores = scores_f + scores_i

        return scores, self.beta * s_loss

    def weighted_GAT(self, tar_embedding_i, neig_embedding_i, adjacency_weight, q1, q2):
        attention_adjacency_mask = torch.where(adjacency_weight != 0, 0, -10000)
        alpha = torch.matmul(torch.cat([tar_embedding_i.unsqueeze(2) * neig_embedding_i, adjacency_weight.unsqueeze(-1)], -1), q1)
        alpha = torch.matmul(self.leakyrelu(alpha), q2).squeeze(-1)
        alpha = alpha + attention_adjacency_mask
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        neig_embedding_i = torch.sum(alpha * neig_embedding_i, 2)
        return neig_embedding_i

    def generate_sess_embedding_f(self, final_embedding_f, mask):
        attention_mask = (1.0 - mask) * -10000.0
        alpha = torch.matmul(final_embedding_f, self.u_feature).squeeze(-1)
        alpha = alpha + attention_mask
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        sess_embedding_f = torch.sum(alpha * final_embedding_f, 1)
        return sess_embedding_f

    def generate_embedding_f(self, embedding_f, re_inputs, mask, max_len, batch_size):
        out_embedding_f = embedding_f[re_inputs]
        embedding_pos_feature = self.f_embedding_pos.weight[:max_len].unsqueeze(0).repeat(batch_size, 1, 1)
        h = self.W2_feature(torch.cat([embedding_pos_feature, out_embedding_f], -1))
        h = torch.tanh(h)
        final_embedding_f = self.multi_head_SSL(h, mask, batch_size, max_len)
        return final_embedding_f

    def discrim_feature(self, x1, x2):
        return torch.sum(x1 * x2, 1)

    def multi_head_SSL(self, h, mask, batch_size, max_len):
        attention_mask = mask.reshape(batch_size, 1, 1, max_len)
        attention_mask = (1.0 - attention_mask) * -10000.0
        mixed_query_layer = self.query(h)
        mixed_key_layer = self.key(h)
        mixed_value_layer = self.value(h)
        query_layer = self.transpose_layer(mixed_query_layer, batch_size, max_len)
        key_layer = self.transpose_layer(mixed_key_layer, batch_size, max_len)
        value_layer = self.transpose_layer(mixed_value_layer, batch_size, max_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = torch.softmax(attention_scores, -1)
        attention_probs = self.dropout20(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)
        sa_result = context_layer.reshape(batch_size, max_len, -1)
        sa_result = sa_result * mask.unsqueeze(-1)
        return sa_result

    def transpose_layer(self, x, batch_size, max_len):
        x = x.reshape(batch_size, max_len, self.num_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

def forward(model, data):
    alias_re_inputs, items, mask, targets, re_inputs, re_groups, re_neg_groups, mask_groups, edge_nodes_co, edge_weight_co, node_edges_co, f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq = data
    alias_re_inputs = trans_to_cuda(alias_re_inputs).long()
    items = trans_to_cuda(items).long()
    mask = trans_to_cuda(mask).long()
    re_inputs = trans_to_cuda(re_inputs).long()
    re_groups = trans_to_cuda(re_groups).long()
    re_neg_groups = trans_to_cuda(re_neg_groups).long()
    mask_groups = trans_to_cuda(mask_groups).long()
    edge_nodes_co = trans_to_cuda(edge_nodes_co).long()
    edge_weight_co = trans_to_cuda(edge_weight_co).float()
    node_edges_co = trans_to_cuda(node_edges_co).long()
    f_adjacency_nodes_seq = trans_to_cuda(f_adjacency_nodes_seq).long()
    f_adjacency_weight_seq = trans_to_cuda(f_adjacency_weight_seq).float()
    b_adjacency_nodes_seq = trans_to_cuda(b_adjacency_nodes_seq).long()
    b_adjacency_weight_seq = trans_to_cuda(b_adjacency_weight_seq).float()
    targets = trans_to_cuda(targets).long()
    scores, s_loss = model(alias_re_inputs, items, mask, re_inputs, re_groups, re_neg_groups, mask_groups, edge_nodes_co, edge_weight_co, node_edges_co, f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq)
    return targets, scores, s_loss