import numpy as np
import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataset import WyzeGraphDataset

class _FakeContext:
    def __init__(self, min_rule_freq=20, resurrect_ghosts=True):
        self.run_config = {
            "min-rule-freq": min_rule_freq,
            "resurrect-ghosts": resurrect_ghosts,
        }

def _per_graph(ds):
    ei, eis = ds.data["edge_index"], ds.slices["edge_index"]
    y, ys = ds.data["y"], ds.slices["y"]
    uid, uids = ds.data["user_id"], ds.slices["user_id"]
    for i in range(ds.num_graphs):
        e = ei[:, eis[i]:eis[i + 1]]
        yy = y[ys[i]:ys[i + 1]]
        u = uid[uids[i]].item() if torch.is_tensor(uid) else uid[i]
        yield int(u), e, yy

if __name__ == "__main__":
    ctx = _FakeContext()
    print("Loading processed caches from data/ ...")
    train = WyzeGraphDataset(context=ctx, mode="train")
    test = WyzeGraphDataset(
        context=ctx, mode="test",
        train_rule2id=train.rule2id, train_device2id=train.device2id,
    )
    print(f"  train graphs={train.num_graphs}  test graphs={test.num_graphs}")
    print(f"  num_rule_types={train.num_rule_types}  num_device_types={train.num_inputs}")

    # Materialize per-graph info once
    train_info = list(_per_graph(train))
    test_info = list(_per_graph(test))
    train_users = {u for u, _, _ in train_info}
    test_users = {u for u, _, _ in test_info}

    # CHECK 1: disjoint test/train (cold start)
    overlap = train_users & test_users
    print("\n[CHECK 1] disjoint test/train (cold start)")
    print(f"  train users={len(train_users)}  test users={len(test_users)}  overlap={len(overlap)}")
    print("  " + ("PASS (fully disjoint)" if not overlap
                   else f"FAIL: {len(overlap)} users in both sets"))

    # CHECK 2: test uses same vocab as train
    same_rules = train.rule2id == test.rule2id
    same_devs = train.device2id == test.device2id
    print("\n[CHECK 2] test uses same vocab as train")
    print(f"  rule2id identical: {same_rules}   device2id identical: {same_devs}")
    print("  " + ("PASS (fully identical)" if (same_rules and same_devs)
                   else "FAIL: test vocab different than train vocab"))

    # CHECK 3: >= rules filter on train but not test
    for name, info in [("train", train_info), ("test", test_info)]:
        counts = np.array([yy.numel() for _, _, yy in info])
        print(f"\n[CHECK 3] {name}: edges (rules) per user")
        print(f"  ==1 rule: {(counts == 1).sum()},  >=2 rules: {(counts >= 2).sum()},"
              f"  median={np.median(counts):.0f}  mean={counts.mean():.2f}  max={counts.max()}")
    print("  if test '==1 rule' is 0, the >=2 filter was incorrectly applied to test")

    # CHECK 4: duplicate triplets per user (test)
    dup_users = 0
    for _, e, yy in test_info:
        if e.size(1) == 0:
            continue
        triples = torch.stack([e[0], e[1], yy], dim=1)
        uniq = torch.unique(triples, dim=0).size(0)
        if uniq < triples.size(0):
            dup_users += 1
    print("\n[CHECK 4] duplicate triplets per user (test)")
    pct = dup_users / max(len(test_info), 1) * 100
    print(f"  test users with >=1 duplicate triplet: {dup_users} ({pct:.1f}%)")

    # CHECK 5: rule-id bounds (must be <num_rule_types)
    max_id = max((int(yy.max()) for _, _, yy in test_info if yy.numel() > 0), default=-1)
    print("\n[CHECK 5] Rule-id bounds")
    print(f"  max rule id={max_id}, num_rule_types={test.num_rule_types} "
          f"→ {'PASS' if max_id < test.num_rule_types else 'FAIL'}")

    print("\nAudit complete.")