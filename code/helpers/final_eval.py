import argparse
import json
import math
import os

import torch
from torch.utils.data import Subset

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import WyzeGraphDataset, train_fit_holdout_indices
from model import build_model, version_from_flags
from task import _rank_heldout_edge

class _FakeContext:
    def __init__(self, min_rule_freq=20, resurrect_ghosts=True):
        self.run_config = {
            "min-rule-freq": min_rule_freq,
            "resurrect-ghosts": resurrect_ghosts,
        }

class _Bucket:
    """Accumulates ranking metrics for a subset of held-out evaluations."""
    def __init__(self, hit_ks):
        self.hit_ks = hit_ks
        self.mr = 0.0
        self.mrr = 0.0
        self.loss = 0.0
        self.hits = {k: 0.0 for k in hit_ks}
        self.n = 0

    def add(self, rank, mrr_cutoff=None, target_prob=None):
        self.mr += rank

        # optional mrr cutoff (2nd place = mrr@50)
        if mrr_cutoff is None or rank <= mrr_cutoff:
            self.mrr += 1.0 / rank
        if target_prob is not None:
            self.loss += -math.log(target_prob + 1e-15)
        for k in self.hit_ks:
            self.hits[k] += 1.0 if rank <= k else 0.0
        self.n += 1

    def summary(self):
        n = max(self.n, 1)
        return {
            "mrr": self.mrr / n,
            "loss": self.loss / n,
            "mr": self.mr / n,
            **{f"hit@{k}": self.hits[k] / n for k in self.hit_ks},
            "num_evals": self.n,
        }

@torch.no_grad()
def final_eval(model, dataset, device, max_users=None, hit_ks=(1, 5, 10), single_edge=False, mrr_cutoff=None):
    """Evaluate either in full LOO mode (single_edge=False) or 2nd place mode (single_edge=True)"""
    model.eval()

    buckets = {name: _Bucket(hit_ks) for name in ("all", "with_context", "no_context")}
    n_users = 0

    ds_len = getattr(dataset, "num_graphs", None) or len(dataset)
    num_graphs = ds_len if max_users is None else min(max_users, ds_len)

    for idx in range(num_graphs):
        graph = dataset[idx].to(device)
        num_edges = graph.num_edges
        if num_edges == 0:
            continue
        n_users += 1

        if single_edge:
            # Deterministic per-user held-out edge (2nd place)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(graph.user_id))
            targets = [torch.randint(0, num_edges, (1,), generator=gen).item()]
        else:
            targets = range(num_edges)

        for target_edge_idx in targets:
            rank, target_prob, _ = _rank_heldout_edge(model, graph, target_edge_idx, device)
            buckets["all"].add(rank, mrr_cutoff, target_prob)
            (buckets["with_context"] if num_edges > 1 else buckets["no_context"]).add(rank, mrr_cutoff, target_prob)

        if (idx + 1) % 500 == 0:
            running = buckets["all"].mrr / max(buckets["all"].n, 1)
            print(f"  ...{idx + 1}/{num_graphs} users, {buckets['all'].n} evals, running MRR={running:.4f}")

    return {
        "num_users": n_users,
        "all": buckets["all"].summary(),
        "with_context": buckets["with_context"].summary(),
        "no_context": buckets["no_context"].summary(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous full leave-one-out evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="test", choices=["test", "train", "holdout"])
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--min-rule-freq", type=int, default=20)
    parser.add_argument("--no-ghosts", action="store_true")
    parser.add_argument("--single-edge", action="store_true")
    parser.add_argument("--mrr-cutoff", type=int, default=None)
    parser.add_argument("--second-place", action="store_true")
    args = parser.parse_args()

    if args.second_place:
        args.single_edge = True
        if args.mrr_cutoff is None:
            args.mrr_cutoff = 50

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    ctx = _FakeContext(min_rule_freq=args.min_rule_freq, resurrect_ghosts=not args.no_ghosts)

    # get vocab from train set
    train = WyzeGraphDataset(context=ctx, mode="train")

    # use test set (cold start)
    if args.mode == "test":
        dataset = WyzeGraphDataset(
            context=ctx, mode="test",
            train_rule2id=train.rule2id, train_device2id=train.device2id,
        )

    # use held-out train set slice
    elif args.mode == "holdout":
        _, holdout_idx = train_fit_holdout_indices(len(train), args.holdout_frac)
        dataset = Subset(train, holdout_idx.tolist())
        print(f"Held-out train slice: {len(dataset)} users (frac={args.holdout_frac})")
    else:
        dataset = train

    # get model version from checkpoint
    rc = {}
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.model)), "config.json")
    try:
        rc = json.load(open(cfg_path))["run_config"]
    except Exception:
        pass
    model_type = str(rc.get("model", "edge"))
    decoder = str(rc.get("decoder", "mlp"))
    encoder = str(rc.get("encoder", "sage"))
    version = version_from_flags(model_type, encoder, decoder)
    print(f"Architecture: {version}"
          + ("" if version == "v1" else f"  (encoder={encoder}, decoder={decoder})"))

    model = build_model(
        version,
        num_node_features=train.num_inputs,
        num_rule_types=train.num_rule_types,
        num_trigger_states=train.num_trigger_states,
        num_action_types=train.num_action_types,
    ).to(device)
    state = torch.load(args.model, map_location=device)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Checkpoint is missing parameters: {result.missing_keys}")
    print(f"Loaded checkpoint: {args.model}")

    protocol = ("2nd-place (single edge, MRR@%d)" % args.mrr_cutoff if args.single_edge and args.mrr_cutoff
                else "single edge" if args.single_edge
                else "full leave-one-out")
    n_users = getattr(dataset, "num_graphs", None) or len(dataset)
    print(f"\nEvaluating on '{args.mode}' ({n_users} users) — protocol: {protocol}...")
    metrics = final_eval(model, dataset, device, max_users=args.max_users,
                            single_edge=args.single_edge, mrr_cutoff=args.mrr_cutoff)

    print(f"\n=== FINAL ({protocol}) ===")
    print(f"  users evaluated: {metrics['num_users']}")
    for bucket in ("all", "with_context", "no_context"):
        b = metrics[bucket]
        print(f"\n  [{bucket}]  (n={b['num_evals']})")
        for k in ("mrr", "loss", "mr", "hit@1", "hit@5", "hit@10"):
            print(f"    {k:8s}: {b[k]:.4f}")

    suffix = "_2ndplace" if (args.single_edge and args.mrr_cutoff) else ("_singleedge" if args.single_edge else "")
    out_dir = os.path.dirname(os.path.abspath(args.model))
    out_path = os.path.join(out_dir, f"final_eval_{args.mode}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.model, "mode": args.mode, "protocol": protocol,
                   "single_edge": args.single_edge, "mrr_cutoff": args.mrr_cutoff,
                   "metrics": metrics}, f, indent=2)
    print(f"\nSaved → {out_path}")
