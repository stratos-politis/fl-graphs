import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

def leave_one_out_split(edge_index, node_batch, device):
    """
    Returns indices for:
    - one held-out prediction edge (pred_indices)
    - the rest, used for message passing (mp_indices)
    """
    edge_graph_ids = node_batch[edge_index[0]]
    num_graphs = edge_graph_ids.max().item() + 1

    mp_indices = []
    pred_indices = []

    for i in range(num_graphs):
        graph_edge_indices = (edge_graph_ids == i).nonzero(as_tuple=True)[0]
        num_graph_edges = graph_edge_indices.size(0)

        if num_graph_edges > 1:
            # Shuffle and pick a random edge to leave out
            shuffled = graph_edge_indices[torch.randperm(num_graph_edges, device=device)]
            mp_indices.append(shuffled[:-1])
            pred_indices.append(shuffled[-1:])
        elif num_graph_edges == 1:
            pred_indices.append(graph_edge_indices)

    if not pred_indices:
        return None, None

    mp_indices = torch.cat(mp_indices) if mp_indices else torch.empty(0, dtype=torch.long, device=device)
    pred_indices = torch.cat(pred_indices)

    return mp_indices, pred_indices

def _sample_negatives_per_user(batch, pos_edge_index, pos_edge_labels, num_rule_types, device, num_negatives=1, hard_frac=0.0):
    """
    Samples K negative (src, dst, rule) triples per positive edge.
    Chooses hard_fract hard negatives (same src, dst, different rule), excluding existing real edges.
    """
    P = pos_edge_index.size(1)
    K = max(1, int(num_negatives))
    if P == 0:
        return (torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device))

    M = P * K
    N = batch.num_nodes

    graph_ids = batch.batch[pos_edge_index[0]].repeat_interleave(K)
    true_rules = pos_edge_labels.repeat_interleave(K)
    ptr = batch.ptr
    lo = ptr[graph_ids]
    hi = ptr[graph_ids + 1]
    span = (hi - lo).clamp(min=1).float()

    # Existing directed edges encoded as unique integer keys for O(1) collision tests.
    existing_keys = batch.edge_index[0] * N + batch.edge_index[1]   # (E,)

    # Hard slots keep the positive's own edge (corrupt rule only); soft slots are resampled.
    K_hard = int(round(K * float(hard_frac)))
    is_hard = (torch.arange(M, device=device) % K) < K_hard          # first K_hard slots per positive
    soft = ~is_hard
    neg_src = pos_edge_index[0].repeat_interleave(K).clone()          # default = positive's edge (hard)
    neg_dst = pos_edge_index[1].repeat_interleave(K).clone()

    remaining = soft.clone()  # only easy/soft slots get resampled to a random within-user edge
    for _ in range(10):  # bounded vectorized rejection; graphs are sparse so this converges fast
        k = int(remaining.sum())
        if k == 0:
            break
        r = remaining
        neg_src[r] = lo[r] + (torch.rand(k, device=device) * span[r]).long()
        neg_dst[r] = lo[r] + (torch.rand(k, device=device) * span[r]).long()
        cand_keys = neg_src * N + neg_dst
        remaining = torch.isin(cand_keys, existing_keys) & soft

    # Negative rule != true rule, in one shot: offset in [1, R-1] guarantees a different class.
    if num_rule_types > 1:
        offset = torch.randint(1, num_rule_types, (M,), device=device)
        neg_rule = (true_rules + offset) % num_rule_types
    else:
        neg_rule = true_rules.clone()

    neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
    return neg_edge_index, neg_rule

def train_client(model, trainloader, epochs, optimizer, device, verbose=False, diagnostic_mode=None, loss_fn="bce", num_negatives=1, hard_frac=0.0):
    """
    Trains local model with leave-one-out splits and a sampled ranking loss.

    Possible loss functions:
    - bce: per-triple binary cross-entropy (true->1, negatives->0).
    - bpr: pairwise -log sigmoid(score_pos - score_neg), averaged over negatives.
    - softmax: sampled-softmax / InfoNCE over {positive, negatives}."""

    loss_fn = str(loss_fn).lower()
    K = max(1, int(num_negatives))
    use_logits = loss_fn in ("softmax", "bpr")   # ranking losses = raw logits, bce = probabilities

    model.train()
    total_loss = 0

    for epoch in range(epochs):
        if verbose:
            progress_bar = tqdm(trainloader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False, position=0)
        else:
            progress_bar = trainloader

        for batch in progress_bar:
            batch = batch.to(device)

            optimizer.zero_grad()

            # 1. leave-one-out split
            mp_idx, pred_idx = leave_one_out_split(batch.edge_index, batch.batch, device)
            if mp_idx is None:
                continue

            pos_edge_index = batch.edge_index[:, pred_idx]
            pos_edge_labels = batch.y[pred_idx]

            if pos_edge_index.size(1) == 0:
                continue

            if diagnostic_mode == 'empty_graph':
                mp_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                mp_edge_attr = torch.empty((0, 2), dtype=torch.long, device=device)
                mp_edge_type = torch.empty((0,), dtype=torch.long, device=device)
            else:
                mp_edge_index = batch.edge_index[:, mp_idx]
                mp_edge_attr = batch.edge_attr[mp_idx]
                mp_edge_type = batch.y[mp_idx]

            num_rule_types = model.num_rule_types

            # 2. negative sampling (K negatives per positive)
            neg_edge_index, neg_edge_labels = _sample_negatives_per_user(
                batch, pos_edge_index, pos_edge_labels, num_rule_types, device,
                num_negatives=K, hard_frac=hard_frac)

            # 3. forward pass
            all_pred_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            out = model(batch.x, mp_edge_index, mp_edge_attr, all_pred_edges,
                        return_logits=use_logits, mp_edge_type=mp_edge_type)

            n_pos = pos_edge_index.size(1)
            pos_scores = torch.gather(out[:n_pos], 1, pos_edge_labels.unsqueeze(1)).squeeze(1)
            neg_scores = torch.gather(out[n_pos:], 1, neg_edge_labels.unsqueeze(1)).squeeze(1)
            neg_scores = neg_scores.view(n_pos, K)

            # 4. ranking loss
            if loss_fn == "softmax":
                logits_all = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                targets = torch.zeros(n_pos, dtype=torch.long, device=device)
                loss = F.cross_entropy(logits_all, targets)

            elif loss_fn == "bpr":
                loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

            else:
                loss = (-torch.log(pos_scores + 1e-15) - torch.log(1.0 - neg_scores + 1e-15).mean(dim=1)).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

            if verbose:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(trainloader.dataset)

def _rank_heldout_edge(model, graph, target_edge_idx, device):
    """
    Holds out one edge and ranks its true triple against all possible permutations.
    Returns (rank, target_prob, total_combinations).
    """
    num_edges = graph.num_edges
    num_nodes = graph.num_nodes

    target_src = graph.edge_index[0, target_edge_idx].item()
    target_dst = graph.edge_index[1, target_edge_idx].item()
    target_rule = graph.y[target_edge_idx].item()

    # Leave-one-out: every edge except the target becomes message-passing
    mp_mask = torch.ones(num_edges, dtype=torch.bool, device=graph.edge_index.device)
    mp_mask[target_edge_idx] = False
    mp_edge_index = graph.edge_index[:, mp_mask]
    mp_edge_attr = graph.edge_attr[mp_mask]
    mp_edge_labels = graph.y[mp_mask]

    # Score every (src, dst) pair
    src_nodes = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    dst_nodes = torch.arange(num_nodes, device=device).repeat(num_nodes)
    all_possible_edges = torch.stack([src_nodes, dst_nodes], dim=0)

    probs = model(graph.x, mp_edge_index, mp_edge_attr, all_possible_edges, mp_edge_type=mp_edge_labels)

    target_row = (target_src * num_nodes) + target_dst
    target_prob = probs[target_row, target_rule]

    # Mask known edges from the candidate set (don't rank)
    if mp_edge_index.size(1) > 0:
        is_target = ((mp_edge_index[0] == target_src) &
                     (mp_edge_index[1] == target_dst) &
                     (mp_edge_labels == target_rule))
        keep = ~is_target
        rows = (mp_edge_index[0] * num_nodes + mp_edge_index[1])[keep]
        cols = mp_edge_labels[keep]
        probs[rows, cols] = 0.0

    # Rank = number of candidates scoring strictly higher than the target +1
    rank = (probs > target_prob).sum().item() + 1.0
    return rank, target_prob.item(), probs.numel()

def eval_client(model, valloader, device, verbose=False):
    """
    Per-round eval: one held-out edge per user, ranked over all possible permutations.
    Outputs MR, MRR, Hit@10, loss.
    """
    model.eval()

    total_loss = 0.0
    total_mr = 0.0
    total_mrr = 0.0
    total_hit10 = 0.0
    num_pos_eval = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="Evaluating", leave=False, position=0) if verbose else valloader

        for batch in progress_bar:
            batch = batch.to(device)

            for graph in batch.to_data_list():
                if graph.num_edges == 0:
                    continue

                # seeded per-user held-out edge
                generator = torch.Generator(device='cpu')
                generator.manual_seed(int(graph.user_id))
                target_edge_idx = torch.randint(0, graph.num_edges, (1,), generator=generator).item()

                rank, target_prob, _ = _rank_heldout_edge(model, graph, target_edge_idx, device)

                total_loss += -np.log(target_prob + 1e-15)
                total_mr += rank
                total_mrr += (1.0 / rank)
                total_hit10 += 1 if rank <= 10 else 0
                num_pos_eval += 1

    n = num_pos_eval if num_pos_eval > 0 else 1
    metrics = {
        "mr": total_mr / n,
        "mrr": total_mrr / n,
        "hit@10": total_hit10 / n,
    }
    return total_loss / n, metrics, num_pos_eval