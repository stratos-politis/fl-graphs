import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class V1Model(nn.Module):
    """v1: GraphSAGE + MLP, no edge types"""

    def __init__(self, num_node_features, num_rule_types, hidden_dim=16):
        super().__init__()

        self.num_rule_types = num_rule_types

        self.conv1 = SAGEConv(num_node_features, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')

        self.pred_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pred_fc2 = nn.Linear(hidden_dim, num_rule_types)

    def forward(self, x, mp_edge_index, mp_edge_attr, pred_edge_index, return_logits=False, mp_edge_type=None):

        h1 = F.relu(self.conv1(x, mp_edge_index))

        z = self.conv2(h1, mp_edge_index)

        edge_features = torch.cat([z[pred_edge_index[0]], z[pred_edge_index[1]]], dim=1)

        logits = self.pred_fc2(F.relu(self.pred_fc1(edge_features)))
        return logits if return_logits else torch.sigmoid(logits)

def _enrich_nodes(model, x, mp_edge_index, mp_edge_attr):
    """
    1) Aggregate edge trigger embeddings --> source node
    2) Aggregate edge action embeddings --> destination node
    3) Concatenate both onto the node one-hot
    """

    num_nodes = x.size(0)

    agg_src = torch.zeros(num_nodes, model.edge_emb_dim, device=x.device)
    agg_dst = torch.zeros(num_nodes, model.edge_emb_dim, device=x.device)

    if mp_edge_index.size(1) > 0:
        agg_src.index_add_(0, mp_edge_index[0], model.trigger_emb(mp_edge_attr[:, 0]))
        agg_dst.index_add_(0, mp_edge_index[1], model.action_emb(mp_edge_attr[:, 1]))

    return torch.cat([x, agg_src, agg_dst], dim=1)

class V2Model(nn.Module):
    """v2: pseudo-embeddings + GraphSAGE + MLP, edge types as node features"""

    def __init__(self, num_node_features, num_rule_types, num_trigger_states, num_action_types, hidden_dim=128, hidden_linear=128, edge_emb_dim=4):
        super().__init__()

        self.num_rule_types = num_rule_types

        self.edge_emb_dim = edge_emb_dim
        self.trigger_emb = nn.Embedding(num_trigger_states, edge_emb_dim)
        self.action_emb = nn.Embedding(num_action_types, edge_emb_dim)
        gnn_input_dim = num_node_features + 2 * edge_emb_dim

        self.conv1 = SAGEConv(gnn_input_dim, hidden_dim * 2, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim * 2, hidden_dim, aggr='mean')

        self.pred_fc1 = nn.Linear(hidden_dim * 2, hidden_linear)
        self.pred_fc2 = nn.Linear(hidden_linear, num_rule_types)

    def forward(self, x, mp_edge_index, mp_edge_attr, pred_edge_index, return_logits=False, mp_edge_type=None):

        x_enriched = _enrich_nodes(self, x, mp_edge_index, mp_edge_attr)

        h1 = F.relu(self.conv1(x_enriched, mp_edge_index))

        z = self.conv2(h1, mp_edge_index)

        edge_features = torch.cat([z[pred_edge_index[0]], z[pred_edge_index[1]]], dim=1)

        logits = self.pred_fc2(F.relu(self.pred_fc1(edge_features)))
        return logits if return_logits else torch.sigmoid(logits)

class CompGCNConv(nn.Module):
    """One CompGCN layer (Vashishth et al, https://arxiv.org/pdf/1911.03082)"""

    def __init__(self, in_dim, out_dim, comp="mult"):
        super().__init__()

        self.comp = str(comp).lower()
        self.out_dim = out_dim

        self.w_o = nn.Linear(in_dim, out_dim, bias=False)    # original direction (u -> v)
        self.w_i = nn.Linear(in_dim, out_dim, bias=False)    # inverse direction  (v -> u)
        self.w_s = nn.Linear(in_dim, out_dim, bias=False)    # self-loop
        self.w_rel = nn.Linear(in_dim, out_dim, bias=False)  # relation update

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _compose(self, h_node, h_rel):
        """phi(h_u, h_r) --> sends to ComplEx"""
        if self.comp == "sub":
            return h_node - h_rel
        return h_node * h_rel

    def forward(self, x, edge_index, edge_type, h_rel):

        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_dim, device=x.device)
        counts = torch.zeros(num_nodes, device=x.device)

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            r = h_rel[edge_type]                                  # [E, in_dim]
            ones = torch.ones(src.size(0), device=x.device)

            # original: compose the source with the relation, send to the destination
            out.index_add_(0, dst, self.w_o(self._compose(x[src], r)))
            counts.index_add_(0, dst, ones)

            # inverse: compose the destination with the relation, send back to the source
            out.index_add_(0, src, self.w_i(self._compose(x[dst], r)))
            counts.index_add_(0, src, ones)

        out = out / counts.clamp(min=1).unsqueeze(1)              # mean over neighbours
        out = out + self.w_s(x) + self.bias                       # self-loop + bias
        return out, self.w_rel(h_rel)

class V3Model(nn.Module):
    """v3: true embeddings + CompGCN + ComplEx, edge types consumed directly by CompGCN"""

    def __init__(self, num_node_features, num_rule_types, num_trigger_states, num_action_types, hidden_dim=128, edge_emb_dim=4, comp_op="mult"):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("V3 (ComplEx) needs an even hidden_dim")

        self.num_rule_types = num_rule_types

        self.hidden_dim = hidden_dim
        self.edge_emb_dim = edge_emb_dim
        self.trigger_emb = nn.Embedding(num_trigger_states, edge_emb_dim)
        self.action_emb = nn.Embedding(num_action_types, edge_emb_dim)
        gnn_input_dim = num_node_features + 2 * edge_emb_dim

        self.rel_base = nn.Embedding(num_rule_types, gnn_input_dim)

        self.comp1 = CompGCNConv(gnn_input_dim, hidden_dim * 2, comp=comp_op)
        self.comp2 = CompGCNConv(hidden_dim * 2, hidden_dim, comp=comp_op)

    def _complex_score(self, z_src, z_dst, rel):

        # Re(<z_s, R_r, conj(z_o)>): split each hidden vector into Re, Im
        d = self.hidden_dim // 2
        a, b = z_src[:, :d], z_src[:, d:]          # Re, Im of z_src
        c, g = z_dst[:, :d], z_dst[:, d:]          # Re, Im of z_dst
        p = a * c + b * g
        q = a * g - b * c
        return p @ rel[:, :d].t() + q @ rel[:, d:].t()

    def forward(self, x, mp_edge_index, mp_edge_attr, pred_edge_index, return_logits=False, mp_edge_type=None):

        x_enriched = _enrich_nodes(self, x, mp_edge_index, mp_edge_attr)

        if mp_edge_type is None:
            mp_edge_type = torch.zeros(mp_edge_index.size(1), dtype=torch.long, device=x.device)

        h_rel = self.rel_base.weight

        h1, h_rel = self.comp1(x_enriched, mp_edge_index, mp_edge_type, h_rel)

        z, h_rel = self.comp2(F.relu(h1), mp_edge_index, mp_edge_type, h_rel)

        logits = self._complex_score(z[pred_edge_index[0]], z[pred_edge_index[1]], h_rel)
        return logits if return_logits else torch.sigmoid(logits)

def build_model(version, num_node_features, num_rule_types, num_trigger_states=None, num_action_types=None):
    """Instantiate a model by version: 'v1', 'v2' or 'v3'."""

    version = str(version).lower()

    if version == "v1":
        return V1Model(num_node_features, num_rule_types)
    if version in ("v2", "v3"):
        cls = V2Model if version == "v2" else V3Model
        return cls(num_node_features, num_rule_types, num_trigger_states, num_action_types)
    raise ValueError(f"Unknown model version {version!r}; expected 'v1', 'v2' or 'v3'")

def version_from_flags(model_type, encoder="sage", decoder="mlp"):
    """Map a run-config's model/encoder/decoder flags to a thesis version.
    Bridges the current config keys to build_model until the run args are cleaned up."""
    if str(model_type).lower() == "simple":
        return "v1"
    return "v3" if str(decoder).lower() == "complex" else "v2"