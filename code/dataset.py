from flwr.common import Context

import torch
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.data.in_memory_dataset import InMemoryDataset

from datasets import load_dataset

import pandas as pd
import numpy as np

import os
import shutil
from tqdm import tqdm

from results_logger import setup_logger
log = setup_logger(__name__, "[DATASET]")

class WyzeGraphDataset(Dataset):
    def __init__(self, context: Context, mode='train', test_set_private=False, train_rule2id=None, train_device2id=None):
        """Loads data from HuggingFace, preprocesses, and precomputes user graphs."""
        log.info(f"Creating global dataset...")

        super().__init__()

        # Load config
        self.min_rule_freq = context.run_config["min-rule-freq"]
        self.resurrect_ghosts = context.run_config["resurrect-ghosts"]
        self.max_train_users = int(context.run_config.get("max-train-users", 0) or 0)

        self.mode = mode
        self.test_set_private = test_set_private

        log.info(f"CONFIG: mode={self.mode}, min_rule_freq={self.min_rule_freq}, resurrect_ghosts={self.resurrect_ghosts}")

        # Set up cache directory for initialization
        cache_dir = "data"
        os.makedirs(cache_dir, exist_ok=True)
        config_tag = f"freq{self.min_rule_freq}_ghost{int(self.resurrect_ghosts)}"
        if mode == 'train' and self.max_train_users > 0:
            config_tag += f"_max{self.max_train_users}"
        self.cache_file = os.path.join(cache_dir, f"{mode}_{config_tag}_graphs_collated.pt")
        self.vocab_file = os.path.join(cache_dir, f"{mode}_{config_tag}_vocab.pt")

        # If cache exists, load tensors directly
        if os.path.exists(self.cache_file) and os.path.exists(self.vocab_file):
            log.info(f"[{self.mode.upper()}] Cache found. Loading from disk...")
            self.data, self.slices = torch.load(self.cache_file, weights_only=False)
            vocab = torch.load(self.vocab_file, weights_only=False)
            self.device2id = vocab['device2id']
            self.rule2id = vocab['rule2id']
            self.num_inputs = vocab['num_inputs']
            self.num_rule_types = vocab['num_rule_types']
            self.num_graphs = vocab['num_graphs']
            nts = vocab.get('num_trigger_states')
            nat = vocab.get('num_action_types')
            if nts is None or nat is None:
                edge_attr = self.data['edge_attr']
                nts = int(edge_attr[:, 0].max()) + 1 if nts is None else nts
                nat = int(edge_attr[:, 1].max()) + 1 if nat is None else nat
            self.num_trigger_states = nts
            self.num_action_types = nat

        # If no cache, process raw HuggingFace data, build graphs, and save to cache
        else:
            log.info(f"[{self.mode.upper()}] No cache found. Processing raw dataset...")

            self._load_data_from_hf()
            self._build_vocab(train_rule2id, train_device2id)
            self._preprocess_dataset()

            self.device_groups = self.df_devices.groupby('user_id')
            self.rule_groups = self.df_rules.groupby('user_id')

            self._build_and_save_all_graphs()

            log.info(f"[{self.mode.upper()}] Saving vocabulary and metadata...")
            torch.save({
                'device2id': self.device2id,
                'rule2id': self.rule2id,
                'num_inputs': self.num_inputs,
                'num_rule_types': self.num_rule_types,
                'num_graphs': self.num_graphs,
                'num_trigger_states': self.num_trigger_states,
                'num_action_types': self.num_action_types
            }, self.vocab_file)

        log.info(f"[{self.mode.upper()}] Dataset ready with {self.num_graphs} users.")

    def _load_data_from_hf(self):
        """Downloads datasets from HuggingFace."""
        log.info(f"Downloading '{self.mode}' data from HuggingFace...")

        if self.mode == 'train':
            device_ds = load_dataset("wyzelabs/RuleRecommendation", data_files="train_device.csv", split="train")
            rule_ds = load_dataset("wyzelabs/RuleRecommendation", data_files="train_rule.csv", split="train")
        elif self.mode == 'test':
            suffix = "_private.csv" if self.test_set_private else ".csv"
            device_ds = load_dataset("wyzelabs/RuleRecommendation", data_files=f"test_device{suffix}", split="train")
            rule_ds = load_dataset("wyzelabs/RuleRecommendation", data_files=f"test_rule{suffix}", split="train")
        else:
            raise ValueError("Mode must be 'train' or 'test'.")

        self.df_devices = device_ds.to_pandas()
        self.df_rules = rule_ds.to_pandas()
        log.info(f"Loaded {len(self.df_devices)} devices and {len(self.df_rules)} rules.")

    def _build_vocab(self, train_rule2id=None, train_device2id=None):
        """Identifies unique devices and rules and assigns unique IDs to them."""

        # Train mode: create vocabulary
        if self.mode == 'train':
            # Device vocabulary
            unique_models = sorted(self.df_devices['device_model'].unique())
            if "Cloud" not in unique_models:        # "Cloud" has rules but is not explicitly in device list
                unique_models.append("Cloud")       # We add it manually
            self.device2id = {name: idx for idx, name in enumerate(unique_models)}

            # Rule vocabulary
            self.df_rules['trigger_action_pair'] = list(zip(self.df_rules['trigger_state'], self.df_rules['action']))
            rule_counts = self.df_rules['trigger_action_pair'].value_counts()
            valid_rules = rule_counts[rule_counts >= self.min_rule_freq].index
            self.rule2id = {rule: idx for idx, rule in enumerate(valid_rules)}

            log.info(f"Built vocab: {len(self.device2id)} unique device models, {len(self.rule2id)} unique rule types.")

        # Test mode: load train vocabulary
        elif self.mode == 'test':
            if train_rule2id is None or train_device2id is None:
                raise ValueError("For test mode, train_rule2id and train_device2id must be provided.")

            # Device vocabulary
            self.device2id = train_device2id

            # Rule vocabulary
            self.df_rules['trigger_action_pair'] = list(zip(self.df_rules['trigger_state'], self.df_rules['action']))
            self.rule2id = train_rule2id

            log.info(f"Loaded train vocab: {len(self.device2id)} devices, {len(self.rule2id)} rules.")

        # Set sizes for model instantiation
        self.num_inputs = len(self.device2id)
        self.num_rule_types = len(self.rule2id)

        # Set sizes for embedding-tables (v2)
        self.num_trigger_states = int(self.df_rules['trigger_state_id'].max()) + 1
        self.num_action_types = int(self.df_rules['action_id'].max()) + 1

    def _preprocess_dataset(self):
        """Filters dataset by pruning infrequent rules and empty graphs."""

        # Prune infrequent rules (appear < min_rule_freq times)
        log.info(f"Pruning infrequent rules...")
        log.info(f"Initial rule count: {len(self.df_rules)}")
        self.df_rules = self.df_rules[self.df_rules['trigger_action_pair'].isin(self.rule2id)]
        log.info(f"Pruned rule count: {len(self.df_rules)}")

        # Prune users with no rules left
        log.info(f"Removing users with no rules left...")
        valid_rule_users = self.df_rules['user_id'].unique()
        self.user_ids = np.intersect1d(self.df_devices['user_id'].unique(), valid_rule_users)

        # Filter users with fewer than 2 rules (train set only)
        if self.mode == 'train':
            rule_counts_per_user = self.df_rules.groupby('user_id').size()
            users_with_2plus_rules = rule_counts_per_user[rule_counts_per_user >= 2].index.values
            self.user_ids = np.intersect1d(self.user_ids, users_with_2plus_rules)
            log.info(f"[TRAIN] Valid users (≥2 rules): {len(self.user_ids)}")

            if self.max_train_users > 0 and len(self.user_ids) > self.max_train_users:
                rng = np.random.RandomState(42)
                chosen = rng.choice(self.user_ids, size=self.max_train_users, replace=False)
                self.user_ids = np.sort(chosen)
                log.info(f"[TRAIN] Subsampled to {len(self.user_ids)} users (max-train-users)")

        # self.user_ids = self.user_ids[:1000]  # small set for fast dev runs

        log.info(f"Total valid users for graph generation: {len(self.user_ids)}")

    def _build_and_save_all_graphs(self):
        """Initially builds all user graphs once at the start of the run."""
        log.info(f"[{self.mode.upper()}] Building PyG graphs...")
        self.num_graphs = len(self.user_ids)
        graphs_list = []

        for idx, user_id in enumerate(tqdm(self.user_ids, desc=f"Generating {self.mode} graphs")):
            # Get users' devices and rules
            try:
                user_devices = self.device_groups.get_group(user_id)
                device_map = dict(zip(user_devices['device_id'], user_devices['device_model']))
            except KeyError:
                device_map = {}

            try:
                user_rules = self.rule_groups.get_group(user_id)
            except KeyError:
                user_rules = pd.DataFrame()

            # Resurrect ghosts: devices referenced in rules but absent from the device list.
            if self.resurrect_ghosts:
                device_map = resurrect_ghosts(device_map, user_rules)

            # Map dataset's long IDs (e.g. 643597) to PyG's sequential node IDs (0, 1...)
            ordered_ids = sorted(device_map.keys())
            local_idx_map = {real_id: i for i, real_id in enumerate(ordered_ids)}

            # Create node features tensor: rows = nodes, cols = device types (e.g. camera, plug)
            num_nodes = len(ordered_ids)
            x = torch.zeros((num_nodes, self.num_inputs), dtype=torch.float)
            for i, real_id in enumerate(ordered_ids):
                model_name = device_map[real_id]
                if model_name in self.device2id:
                    x[i, self.device2id[model_name]] = 1.0

            # Create edges and their labels
            src, dst, feats, labels = [], [], [], []

            if not user_rules.empty and num_nodes > 0:
                for _, row in user_rules.iterrows():
                    u, v = row['trigger_device_id'], row['action_device_id']
                    if u in local_idx_map and v in local_idx_map:
                        src.append(local_idx_map[u])
                        dst.append(local_idx_map[v])
                        feats.append([row['trigger_state_id'], row['action_id']])
                        labels.append(self.rule2id[row['trigger_action_pair']])

            # Return data
            if src:
                graph = Data(
                    x=x,
                    edge_index=torch.tensor([src, dst], dtype=torch.long),
                    edge_attr=torch.tensor(feats, dtype=torch.long),
                    y=torch.tensor(labels, dtype=torch.long),
                    user_id=user_id
                )
            else:
                graph = Data(
                    x=x,
                    edge_index=torch.empty((2,0), dtype=torch.long),
                    edge_attr=torch.empty((0,2), dtype=torch.long),
                    y=torch.empty((0,), dtype=torch.long),
                    user_id=user_id
                )

            graphs_list.append(graph)

        # Collate all graphs for fast memory access
        log.info(f"[{self.mode.upper()}] Collating graphs to minimize RAM footprint...")
        self.data, self.slices = InMemoryDataset.collate(graphs_list)
        torch.save((self.data, self.slices), self.cache_file)

        # Delete garbage after finishing
        del self.df_devices
        del self.df_rules
        del self.device_groups
        del self.rule_groups
        del graphs_list
        import gc
        gc.collect()

    def get(self, idx):
        """Slices requested graph out of the collated tensors in RAM."""
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx], slices[idx + 1]

            if key == 'edge_index':
                data[key] = item[:, start:end]

            elif key == 'user_id':
                data[key] = item[start].item() if torch.is_tensor(item) else item[start]

            else:
                data[key] = item[start:end]

        return data

    def len(self):
        return self.num_graphs

def resurrect_ghosts(device_map, user_rules):
    """Add devices referenced in a user's rules but missing from their device list."""
    for _, row in user_rules.iterrows():
        if row['trigger_device_id'] not in device_map:
            device_map[row['trigger_device_id']] = row['trigger_device']
        if row['action_device_id'] not in device_map:
            device_map[row['action_device_id']] = row['action_device']
    return device_map

def train_fit_holdout_indices(total_size, holdout_frac, seed=42):
    """
    Reproducible IID split of user indices into (fit, holdout) for evaluation.
    Used for 2nd place's eval instead of separate test set.
    """

    indices = np.arange(total_size)

    # same seed as partitioning
    np.random.RandomState(seed).shuffle(indices)

    n_holdout = int(total_size * holdout_frac)
    if n_holdout <= 0:
        return indices, np.array([], dtype=int)

    # reserve last holdout_frac as eval slice
    return indices[:total_size - n_holdout], indices[total_size - n_holdout:]

def prepare_client_dataloader(dataset, partition_id, num_partitions, batch_size=32, shuffle=True, pin_memory=True, holdout_frac=0.0):
    """Partitions the global dataset into non-overlapping subsets and returns a PyG DataLoader for the specific client."""
    log.info("Partitioning dataset ...")

    total_size = len(dataset)

    # Fit pool = all users minus reserved holdout slice (same seed everywhere)
    fit_indices, _ = train_fit_holdout_indices(total_size, holdout_frac)

    # Split fit pool into 'num_partitions' chunks
    partitions = np.array_split(fit_indices, num_partitions)
    client_indices = partitions[partition_id].tolist()

    client_subset = Subset(dataset, client_indices)
    return DataLoader(client_subset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

def prepare_holdout_dataloader(dataset, holdout_frac, batch_size=32, pin_memory=True):
    """DataLoader over the reserved eval slice."""
    total_size = len(dataset)

    _, holdout_indices = train_fit_holdout_indices(total_size, holdout_frac)

    holdout_subset = Subset(dataset, holdout_indices.tolist())
    return DataLoader(holdout_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)