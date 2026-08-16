import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from model import V2Model
from task import leave_one_out_split, eval_client, train_client

# Helpers
def _make_synthetic_batch(num_graphs=3, num_nodes_per_graph=4, edges_per_graph=4, num_rule_types=10, device='cpu'):
    """Makes dummy PyG tensors for testing"""
    graphs = []
    for g in range(num_graphs):
        n = num_nodes_per_graph
        x = torch.zeros(n, 16)
        x[torch.arange(n), torch.randint(0, 16, (n,))] = 1.0

        src = torch.randint(0, n, (edges_per_graph,))
        dst = torch.randint(0, n, (edges_per_graph,))
        edge_index = torch.stack([src, dst], dim=0)
        y = torch.randint(0, num_rule_types, (edges_per_graph,))
        edge_attr = torch.stack([
            torch.randint(0, 45, (edges_per_graph,)),
            torch.randint(0, 47, (edges_per_graph,))
        ], dim=1)

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                           y=y, user_id=1000 + g))
    return Batch.from_data_list(graphs).to(device)

def _make_model(num_rule_types=10, num_trigger_states=45, num_action_types=47, device='cpu'):
    """Creates a model to train on the dummy graphs"""
    model = V2Model(
        num_node_features=16,
        num_rule_types=num_rule_types,
        num_trigger_states=num_trigger_states,
        num_action_types=num_action_types,
        hidden_dim=8,
        hidden_linear=8,
        edge_emb_dim=4
    ).to(device)
    return model

def _ok(name):
    """Pass message"""
    print(f"  [PASS] {name}")

def _fail(name, msg):
    """Fail and exit"""
    print(f"  [FAIL] {name}: {msg}")
    sys.exit(1)

# Test 1: leave_one_out_split correctness
def test_leave_one_out_split():
    device = 'cpu'
    batch = _make_synthetic_batch(num_graphs=3, num_nodes_per_graph=4, edges_per_graph=4)
    total_edges = batch.edge_index.size(1)

    mp_idx, pred_idx = leave_one_out_split(batch.edge_index, batch.batch, device)
    if mp_idx is None:
        _fail("leave_one_out_split", "returned None for a valid batch")

    # should leave out exactly 1 edge per graph (3 graphs -> 3 edges)
    n_sup = pred_idx.size(0)
    if n_sup != 3:
        _fail("leave_one_out_split",
              f"expected 3 supervision edges (1 per graph), got {n_sup}")

    # mp + pred should cover all edges, with no overlap
    n_mp = mp_idx.size(0)
    if n_mp + n_sup != total_edges:
        _fail("leave_one_out_split",
              f"mp ({n_mp}) + pred ({n_sup}) != total ({total_edges})")
    overlap = set(mp_idx.tolist()) & set(pred_idx.tolist())
    if overlap:
        _fail("leave_one_out_split", f"mp and pred indices overlap: {overlap}")

    _ok("leave_one_out_split basic split (3 graphs)")

    # edge case: graph with 1 edge only -> that edge is the prediction, 0 mp edges
    single_edge_data = Data(
        x=torch.eye(4),
        edge_index=torch.tensor([[0], [1]]),
        y=torch.tensor([0]),
        edge_attr=torch.tensor([[0, 0]]),
        user_id=999
    )
    batch_single = Batch.from_data_list([single_edge_data])
    mp2, pred2 = leave_one_out_split(batch_single.edge_index, batch_single.batch, device)
    if mp2 is None:
        _fail("leave_one_out_split", "returned None for a single-edge graph batch")
    if pred2.size(0) != 1:
        _fail("leave_one_out_split",
              f"single-edge graph: expected 1 pred edge, got {pred2.size(0)}")
    if mp2.size(0) != 0:
        _fail("leave_one_out_split",
              f"single-edge graph: expected 0 mp edges, got {mp2.size(0)}")

    _ok("leave_one_out_split single-edge graph")

# Test 2: eval_client ranking sanity
def test_eval_ranking_sanity():
    from torch_geometric.loader import DataLoader

    num_nodes = 4
    num_rules = 10

    # graph with 4 edges, all different rule types
    x = torch.eye(num_nodes)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    y = torch.tensor([2, 5, 7, 9])
    user_id = 42
    graph = Data(x=x, edge_index=edge_index, y=y,
                 edge_attr=torch.zeros(4, 2, dtype=torch.long),
                 user_id=user_id)
    loader = DataLoader([graph], batch_size=1)

    # eval_client chooses edge based on seed, pre-compute which one it will choose
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(user_id))
    target_idx = torch.randint(0, 4, (1,), generator=gen).item()
    target_src = edge_index[0, target_idx].item()
    target_dst = edge_index[1, target_idx].item()
    target_rule = y[target_idx].item()
    target_row = target_src * num_nodes + target_dst
    total_combos = num_nodes * num_nodes

    # best case: target gets probability=1, rest=0
    best_probs = torch.zeros(total_combos, num_rules)
    best_probs[target_row, target_rule] = 1.0

    class BestModel(nn.Module):
        def eval(self): return self
        def __call__(self, x, mp_ei, mp_el, pred_ei, **kwargs):
            return best_probs[:pred_ei.size(1)].clone()
        def parameters(self): return iter([])

    best_model = BestModel()
    _, metrics, _ = eval_client(best_model, loader, device='cpu')
    if abs(metrics['mrr'] - 1.0) > 1e-6:
        _fail("eval ranking (best case)",
              f"mrr should be 1.0, got {metrics['mrr']} "
              f"(target: src={target_src} dst={target_dst} rule={target_rule} row={target_row})")
    if metrics['mr'] != 1.0:
        _fail("eval ranking (best case)", f"mr should be 1, got {metrics['mr']}")
    if metrics['hit@10'] != 1.0:
        _fail("eval ranking (best case)", f"hit@10 should be 1.0, got {metrics['hit@10']}")

    _ok("eval_client rank sanity — best case (rank=1)")

    # worst-case: target gets 0, rest=1
    worst_probs = torch.ones(total_combos, num_rules)
    worst_probs[target_row, target_rule] = 0.0

    class WorstModel(nn.Module):
        def eval(self): return self
        def __call__(self, x, mp_ei, mp_el, pred_ei, **kwargs):
            return worst_probs[:pred_ei.size(1)].clone()
        def parameters(self): return iter([])

    worst_model = WorstModel()
    _, metrics, _ = eval_client(worst_model, loader, device='cpu')
    if metrics['hit@10'] != 0.0:
        _fail("eval ranking (worst case)", f"hit@10 should be 0.0, got {metrics['hit@10']}")

    # target should rank near the bottom (huge mr, tiny mrr)
    if metrics['mrr'] > 0.05:
        _fail("eval ranking (worst case)", f"mrr should be ~0 (< 0.05), got {metrics['mrr']}")

    _ok("eval_client rank sanity — worst case (rank=last)")

# Test 3: model produces different outputs with vs without edge context
def test_edge_context_changes_output():
    device = 'cpu'
    num_rules = 10
    model = _make_model(num_rule_types=num_rules, device=device)
    model.eval()

    num_nodes = 5
    x = torch.zeros(num_nodes, 16)
    x[torch.arange(num_nodes), torch.randint(0, 16, (num_nodes,))] = 1.0
    mp_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    mp_edge_attr = torch.tensor([[3, 4], [5, 6], [7, 8]])
    pred_edge_index = torch.tensor([[0, 1], [1, 2]])

    with torch.no_grad():
        # model with message passing conetxt
        probs_full = model(x, mp_edge_index, mp_edge_attr, pred_edge_index).clone()

        # model with no message passing context
        empty_ei = torch.empty((2, 0), dtype=torch.long)
        empty_attr = torch.empty((0, 2), dtype=torch.long)
        probs_empty = model(x, empty_ei, empty_attr, pred_edge_index).clone()

    if torch.allclose(probs_full, probs_empty):
        _fail("edge context changes output",
              "model outputs identical results with and without edge context — "
              "edge embeddings are not flowing into predictions")

    _ok("model output changes with vs without edge context")

# Test 4: loss decreases over 5 gradient steps
def test_loss_decreases():
    device = 'cpu'

    # single short run is noisy, fix seed and smooth over many runs
    torch.manual_seed(0)

    num_rules = 5
    model = _make_model(num_rule_types=num_rules, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)

    from torch_geometric.loader import DataLoader
    graphs = []
    for u in range(4):
        n = 6
        x = torch.zeros(n, 16)
        x[torch.arange(n), torch.arange(n)] = 1.0
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([1, 2, 3, 4, 5])
        edge_index = torch.stack([src, dst])

        # deterministic learnable rule
        y = (src + dst) % num_rules

        edge_attr = torch.zeros(5, 2, dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr, user_id=u))
    loader = DataLoader(graphs, batch_size=4)

    # train for 30 steps and compare mean of first 5 vs mean of last 5 to smooth noise
    losses = []
    for _ in range(30):
        losses.append(train_client(model, loader, epochs=1, optimizer=optimizer, device=device, diagnostic_mode=None))

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    if late >= early:
        _fail("loss decreases over training",
              f"smoothed loss did not decrease: early={early:.4f} -> late={late:.4f}")

    _ok(f"loss decreases over 30 steps (early={early:.4f} -> late={late:.4f})")

# Test 5: end to end run (neg sampling and loss test)
def test_per_user_negative_sampling():
    device = 'cpu'
    num_rules = 10

    from torch_geometric.loader import DataLoader
    batch = _make_synthetic_batch(num_graphs=3, num_nodes_per_graph=4, edges_per_graph=4, num_rule_types=num_rules)
    loader = DataLoader(batch.to_data_list(), batch_size=3)

    model = _make_model(num_rule_types=num_rules, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    try:
        loss = train_client(model, loader, epochs=1, optimizer=optimizer, device=device, diagnostic_mode=None)
        if loss != loss:  # NaN check
            _fail("per-user negative sampling", "loss is NaN")
    except Exception as e:
        _fail("per-user negative sampling", f"training raised exception: {e}")

    _ok("negative sampling runs without error")

# Test 6: prove pseudo-embeddings help
def test_empty_vs_full_graph_after_training():
    from torch_geometric.loader import DataLoader

    device = 'cpu'
    num_rules = 5
    torch.manual_seed(0)

    # 10 users, assign type r_u in [0, num_rules)
    # carry that type in y and edge_attr but shared node identities
    # so it can only be learnable by edge information

    graphs = []
    for i in range(10):
        n = 6
        r_u = i % num_rules
        x = torch.eye(16)[:n]
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        y = torch.full((5,), r_u)
        edge_attr = torch.full((5, 2), r_u, dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y,
                           edge_attr=edge_attr, user_id=i + 100))
    loader = DataLoader(graphs, batch_size=10)

    def train_fresh(diag):
        torch.manual_seed(1)  # identical init for fair comparison
        model = _make_model(num_rule_types=num_rules)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(40):
            train_client(model, loader, epochs=1, optimizer=opt,
                         device=device, diagnostic_mode=diag)
        return model

    model_empty = train_fresh('empty_graph')
    model_full = train_fresh(None)

    eval_loader = DataLoader(graphs, batch_size=10, shuffle=False)
    _, metrics_empty, _ = eval_client(model_empty, eval_loader, device=device)
    _, metrics_full, _ = eval_client(model_full, eval_loader, device=device)

    print(f"    MRR empty_graph={metrics_empty['mrr']:.4f}  full_graph={metrics_full['mrr']:.4f}")

    if metrics_full['mrr'] <= metrics_empty['mrr']:
        _fail("edge context helps learning",
              f"full ({metrics_full['mrr']:.4f}) should beat empty "
              f"({metrics_empty['mrr']:.4f}) — edge-attr context not being exploited")

    _ok("full-graph training beats empty (trigger/action context is exploited)")

# Test 7: duplicate edge must not mask the held-out target
def test_duplicate_edge_not_masked():
    from task import _rank_heldout_edge
    from torch_geometric.data import Data

    num_nodes, num_rules = 4, 10

    # identical edges 0 and 1
    edge_index = torch.tensor([[0, 0, 2], [1, 1, 3]])
    y = torch.tensor([5, 5, 7])
    x = torch.eye(num_nodes)
    graph = Data(x=x, edge_index=edge_index, y=y,
                 edge_attr=torch.zeros(3, 2, dtype=torch.long), user_id=1)

    # constant model
    M = torch.full((num_nodes * num_nodes, num_rules), 0.1)
    M[(0 * num_nodes) + 1, 5] = 0.9

    class ConstModel(nn.Module):
        def eval(self): return self
        def __call__(self, x, mp_ei, mp_el, pred_ei, **kwargs):
            return M[:pred_ei.size(1)].clone()

    # hold out edge 0, edge 1 must be on list
    rank, target_prob, _ = _rank_heldout_edge(ConstModel(), graph, 0, device='cpu')

    if abs(target_prob - 0.9) > 1e-5:
        _fail("duplicate edge not masked",
              f"target prob should be preserved at ~0.9, got {target_prob} "
              f"(duplicate edge masked the target)")
    if rank != 1.0:
        _fail("duplicate edge not masked", f"target should rank 1, got {rank}")

    _ok("duplicate (src,dst,rule) edge does not mask the held-out target")


if __name__ == '__main__':
    print("=" * 60)
    print("  Running tests...")
    print("=" * 60)

    print("\n[1] leave_one_out_split correctness")
    test_leave_one_out_split()

    print("\n[2] eval_client ranking sanity")
    test_eval_ranking_sanity()

    print("\n[3] model produces different outputs with vs without edge context")
    test_edge_context_changes_output()

    print("\n[4] loss decreases over 5 gradient steps")
    test_loss_decreases()

    print("\n[5] end to end run (neg sampling and loss test)")
    test_per_user_negative_sampling()

    print("\n[6] prove pseudo-embeddings help")
    test_empty_vs_full_graph_after_training()

    print("\n[7] duplicate edge must not mask the held-out target")
    test_duplicate_edge_not_masked()

    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)