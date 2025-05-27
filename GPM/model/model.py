import copy
import os.path as osp
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from utils.eval import *
from utils.utils import get_device_from_model, check_path
from .encoder import PatternEncoder
from .vq import VectorQuantize

from torch_geometric.nn import Node2Vec


# From https://github.com/yuanqing-wang/rum
class Consistency(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, probs):
        avg_probs = probs.mean(0)
        sharpened_probs = avg_probs.pow(1 / self.temperature)
        sharpened_probs = sharpened_probs / sharpened_probs.sum(-1, keepdim=True)
        loss = (sharpened_probs - avg_probs).pow(2).sum(-1).mean()
        return loss


# From https://github.com/yuanqing-wang/rum
class SelfSupervise(torch.nn.Module):
    def __init__(self, in_features, out_features, subsample=100, binary=True):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.subsample = subsample
        self.binary = binary

    def forward(self, y_hat, y):
        idxs = torch.randint(high=y_hat.shape[-3], size=(self.subsample,), device=y.device)
        y, y_hat = y.flatten(0, -3), y_hat.flatten(0, -3)
        y = y[..., idxs, 1:, :].contiguous()
        y_hat = y_hat[..., idxs, :-1, :].contiguous()
        y_hat = self.fc(y_hat)
        if self.binary:
            loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=y.detach().mean().pow(-1)
            )(y_hat, y)
        else:
            # loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            loss = torch.nn.MSELoss()(y_hat, y)
        return loss


# Adapted from official ogb implementation
# https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ppa/gnn.py
class LinkPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(torch.nn.Linear(hidden_dim, output_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.input_dim = params['input_dim'] + params['edge_dim'] + params['node_pe_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.num_layers = params['num_layers']
        self.pattern_encoder = PatternEncoder(params)

        self.vq = VectorQuantize(
            dim=self.hidden_dim,
            codebook_size=params["codebook_size"],
            codebook_dim=self.hidden_dim,
            heads=params['heads'],
            separate_codebook_per_head=True,
            use_cosine_sim=True,
            kmeans_init=True,
            ema_update=True,
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=params["heads"],
            dim_feedforward=self.hidden_dim * 4,
            dropout=params["dropout"],
            norm_first=params["norm_first"]
        )
        self.encoder = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(self.hidden_dim)

        if params['task'] in ['node', 'graph']:
            self.head = nn.Linear(self.hidden_dim, self.output_dim)
        elif params['task'] in ['link']:
            self.head = LinkPredictor(self.hidden_dim, self.hidden_dim, 1, 3, 0.0)

        if params['use_attn_fusion']:
            attn_dim = self.hidden_dim if not params['use_cls_token'] else 2 * self.hidden_dim
            self.attn_layer = nn.Linear(attn_dim, 1)

    def linear_probe(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def transformer_encode(self, x):
        for layer in self.encoder:
            last_x = x
            x = layer(self.norm(x))
            x = last_x + x
        return x

    def get_instance_emb(self, pattern_emb, params):
        if params['use_cls_token']:
            if params['use_attn_fusion']:
                target = pattern_emb[0, :, :]  # [n, d]
                source = pattern_emb[1:, :, :]  # [h-1, n, d]
                attn = self.attn_layer(
                    torch.cat([target.unsqueeze(0).repeat(source.size(0), 1, 1), source], dim=-1)).squeeze(-1)
                attn = F.softmax(attn, dim=0)
                instance_emb = torch.sum(attn.unsqueeze(-1) * source, dim=0) + target
            else:
                instance_emb = pattern_emb[0].squeeze(0)
        else:
            if params['use_attn_fusion']:
                source = pattern_emb
                attn = self.attn_layer(source).squeeze(-1)
                attn = F.softmax(attn, dim=0)
                instance_emb = torch.sum(attn.unsqueeze(-1) * source, dim=0)
            else:
                instance_emb = pattern_emb.mean(dim=0)
        return instance_emb

    def forward(self, graph, items, params, **kwargs):
        mode = kwargs['mode']
        if params['task'] == 'node':
            return self.encode_node(graph, items, params, mode)
        elif params['task'] == 'link':
            return self.encode_link(graph, items, params, mode)
        elif params['task'] == 'graph':
            return self.encode_graph(graph, items, params, mode)
        else:
            raise ValueError(f"Unsupported task: {params['task']}")

    def encode_node(self, graph, nodes, params, mode):
        device = get_device_from_model(self)

        feat = graph.x
        node_pe = graph.pe if graph.get('pe') is not None else None

        # Get patterns
        num_patterns = params['num_patterns']
        pattern_set = params['pattern_set']

        # Get patterns for target nodes
        all_patterns = pattern_set['pattern']
        selected_patterns = all_patterns[:, nodes, :]
        h, num_nodes, k = selected_patterns.shape

        # Randomly select patterns during training
        if mode == 'train':
            idx = torch.randint(0, h, (num_nodes, num_patterns))
            patterns = torch.stack([selected_patterns[idx[i], i, :] for i in range(num_nodes)], dim=1)
        else:
            patterns = selected_patterns

        if graph.edge_attr is not None:
            e_feat = graph.edge_attr
            all_eid = pattern_set['eid']
            selected_eid = all_eid[:, nodes, :]
            if mode == 'train':
                eids = torch.stack([selected_eid[idx[i], i, :] for i in range(num_nodes)], dim=1)
                e_feat = e_feat[eids].to(device)
            else:
                eids = selected_eid
                e_feat = e_feat[eids].to(device)
        else:
            e_feat = None

        pattern_feat = self.pattern_encoder.encode_node(patterns, feat, node_pe, e_feat, params)

        if params['use_vq']:
            pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
        else:
            commit_loss = 0

        if params['use_cls_token']:
            cls_token = torch.ones(1, pattern_feat.size(1), pattern_feat.size(2), device=device)
            pattern_feat = torch.cat([cls_token, pattern_feat], dim=0)
        pattern_emb = self.transformer_encode(pattern_feat)
        instance_emb = self.get_instance_emb(pattern_emb, params)

        pred = self.head(instance_emb)

        return pred, instance_emb, pattern_emb, commit_loss

    def encode_link(self, graph, links, params, mode):
        device = get_device_from_model(self)

        feat = graph.x
        node_pe = graph.pe if graph.get('pe') is not None else None

        source_nodes, target_nodes = links[:, 0], links[:, 1]
        all_nodes = {'source': source_nodes, 'target': target_nodes}
        edge_emb = {'pattern': {}, 'instance': {}, 'commit_loss': {}}

        for key, nodes in all_nodes.items():
            # Get patterns
            num_patterns = params['num_patterns']
            pattern_set = params['pattern_set']

            # Get patterns for target nodes
            all_patterns = pattern_set['pattern']
            selected_patterns = all_patterns[:, nodes, :]
            h, num_nodes, k = selected_patterns.shape

            # Randomly select patterns during training
            if mode == 'train':
                idx = torch.randint(0, h, (num_nodes, num_patterns))
                patterns = torch.stack([selected_patterns[idx[i], i, :] for i in range(num_nodes)], dim=1)
            else:
                patterns = selected_patterns

            if graph.edge_attr is not None:
                e_feat = graph.edge_attr
                all_eid = pattern_set['eid']
                selected_eid = all_eid[:, nodes, :]
                if mode == 'train':
                    eids = torch.stack([selected_eid[idx[i], i, :] for i in range(num_nodes)], dim=1)
                    e_feat = e_feat[eids].to(device)
                else:
                    eids = selected_eid
                    e_feat = e_feat[eids].to(device)
            else:
                e_feat = None

            pattern_feat = self.pattern_encoder.encode_node(patterns, feat, node_pe, e_feat, params)

            if params['use_vq']:
                pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
            else:
                commit_loss = 0

            if params['use_cls_token']:
                cls_token = torch.ones(1, pattern_feat.size(1), pattern_feat.size(2), device=device)
                pattern_feat = torch.cat([cls_token, pattern_feat], dim=0)
            pattern_emb = self.transformer_encode(pattern_feat)
            instance_emb = self.get_instance_emb(pattern_emb, params)

            edge_emb['pattern'][key] = pattern_emb
            edge_emb['instance'][key] = instance_emb
            edge_emb['commit_loss'][key] = commit_loss

        instance_emb = edge_emb['instance']['source'] * edge_emb['instance']['target']
        pattern_emb = torch.cat([edge_emb['pattern']['source'], edge_emb['pattern']['target']], dim=-1)
        commit_loss = edge_emb['commit_loss']['source'] + edge_emb['commit_loss']['target']

        pred = self.head(instance_emb)

        return pred, instance_emb, pattern_emb, commit_loss

    def encode_graph(self, graph, graphs, params, mode):
        device = get_device_from_model(self)

        feat = graph._data.x_feat.to(device)

        # Get patterns
        num_patterns = params['num_patterns']
        pattern_set = params['pattern_set']
        if pattern_set.get('train') is not None:
            pattern_set = pattern_set[mode] if params['split'] != 'pretrain' else pattern_set['full']

        # Get patterns for target graphs
        all_patterns = pattern_set['pattern']
        all_nid = pattern_set['nid']
        selected_patterns = all_patterns[:, graphs, :]
        selected_nid = all_nid[:, graphs, :]
        h, num_graphs, k = selected_nid.shape

        # In training, selecting a subset of patterns
        if mode == 'train':
            idx = torch.randint(0, h, (num_graphs, num_patterns))
            patterns = torch.stack([selected_patterns[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
            nids = torch.stack([selected_nid[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
        else:
            patterns = selected_patterns.to(device)
            nids = selected_nid.to(device)

        if graph[0].get('pe') is not None:
            node_pe_list = [graph[g].pe for g in graphs]
            max_nodes = max(pe.size(0) for pe in node_pe_list)
            dim = node_pe_list[0].size(1)
            node_pe = torch.zeros((len(node_pe_list), max_nodes, dim))
            for i, pe in enumerate(node_pe_list):
                node_pe[i, :pe.size(0), :] = pe
            node_pe = node_pe.to(device)
        else:
            node_pe = None

        if graph._data.edge_attr is not None:
            e_feat = graph._data.e_feat.to(device)
            all_eid = pattern_set['eid']
            selected_eid = all_eid[:, graphs, :]
            if mode == 'train':
                eids = torch.stack([selected_eid[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
            else:
                eids = selected_eid.to(device)
        else:
            e_feat = None
            eids = None

        pattern_feat = self.pattern_encoder.encode_graph(nids, feat, patterns, eids, e_feat, node_pe, params)

        if params['use_vq']:
            pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
        else:
            commit_loss = 0

        if params['use_cls_token']:
            cls_token = torch.ones(1, pattern_feat.size(1), pattern_feat.size(2), device=device)
            pattern_feat = torch.cat([cls_token, pattern_feat], dim=0)
        pattern_emb = self.transformer_encode(pattern_feat)
        instance_emb = self.get_instance_emb(pattern_emb, params)

        pred = self.head(instance_emb)

        return pred, instance_emb, pattern_emb, commit_loss
