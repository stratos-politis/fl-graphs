import os
import sys
import csv
import json
import logging
import platform
from datetime import datetime

from flwr.server.strategy import FedAvg, FedAdam
from flwr.common import parameters_to_ndarrays

def setup_logger(name, prefix):
    """Bypass flwr's logger, only print once"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(f"{prefix} %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def _versions():
    """version metadata to include in logs"""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("flwr", "torch", "torch_geometric", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions

class ResultsRecorder:
    def __init__(self, base_dir="results", run_config=None, extra_meta=None, resume_dir=None, round_offset=0, checkpoint_every=1):

        # round offset for resumed runs
        self.round_offset = int(round_offset or 0)

        # when to checkpoint (default every round)
        self.checkpoint_every = max(1, int(checkpoint_every or 1))

        self._data = {"train": {}, "centralized": {}, "distributed": {}}

        # track best model
        self._best_value = None
        self._best_round = None

        # resume run
        if resume_dir:
            self.run_dir = resume_dir
            self.timestamp = os.path.basename(resume_dir.rstrip("/")).replace("run_", "")
            self._load_existing()

        # new run
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(base_dir, f"run_{timestamp}")
            os.makedirs(self.run_dir, exist_ok=True)
            self.timestamp = timestamp
            self._write_config(run_config or {}, extra_meta or {})

    def _load_existing(self):
        """fill in previous rounds and best checkpoint for resumed run"""
        try:
            with open(self.metrics_json_path) as f:
                out = json.load(f)
            for split, series in out.items():
                for metric, pairs in series.items():
                    for rnd, value in pairs:
                        self._data.setdefault(split, {}).setdefault(int(rnd), {})[metric] = value
        except Exception:
            pass
        try:
            with open(os.path.join(self.run_dir, "best_checkpoint.json")) as f:
                bc = json.load(f)
            self._best_round = bc.get("best_round")
            for k, v in bc.items():
                if k.startswith("best_") and k != "best_round":
                    self._best_value = v
        except Exception:
            pass

    # grab paths from running dir
    @property
    def config_path(self):
        return os.path.join(self.run_dir, "config.json")

    @property
    def metrics_json_path(self):
        return os.path.join(self.run_dir, "metrics.json")

    @property
    def metrics_csv_path(self):
        return os.path.join(self.run_dir, "metrics.csv")

    @property
    def best_model_path(self):
        return os.path.join(self.run_dir, "best_model.pt")

    @property
    def latest_checkpoint_path(self):
        return os.path.join(self.run_dir, "latest_checkpoint.pt")

    def save_latest(self, abs_round, params_ndarrays, adam_state=None):
        """save latest checkpoint with global params"""
        import torch
        payload = {
            "round": int(abs_round),
            "params": [p for p in params_ndarrays],
            "best_value": self._best_value,
            "best_round": self._best_round,
        }
        if adam_state is not None:
            payload["m_t"], payload["v_t"] = adam_state
        tmp = self.latest_checkpoint_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.latest_checkpoint_path)

    def maybe_save_best(self, server_round, metric_value, state_dict, metric_name="mrr", higher_is_better=True):
        """save best checkpoint if it improves metric (default mrr)"""
        if metric_value is None:
            return False

        improved = (
            self._best_value is None
            or (metric_value > self._best_value if higher_is_better
                else metric_value < self._best_value)
        )
        if not improved:
            return False

        abs_round = server_round + self.round_offset
        self._best_value = metric_value
        self._best_round = abs_round

        import torch
        cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        torch.save(cpu_state, self.best_model_path)

        with open(os.path.join(self.run_dir, "best_checkpoint.json"), "w") as f:
            json.dump({
                "selection_metric": metric_name,
                "best_round": abs_round,
                f"best_{metric_name}": metric_value,
            }, f, indent=2)
        return True

    def _write_config(self, run_config, extra_meta):
        """write config"""
        snapshot = {
            "timestamp": self.timestamp,
            "versions": _versions(),
            "run_config": {k: _jsonable(v) for k, v in dict(run_config).items()},
            **{k: _jsonable(v) for k, v in extra_meta.items()},
        }
        with open(self.config_path, "w") as f:
            json.dump(snapshot, f, indent=2)

    def log_train_loss(self, server_round, loss):
        """log round's train loss"""
        r = server_round + self.round_offset
        self._data["train"].setdefault(r, {})["loss"] = float(loss)
        self._flush()

    def log_centralized(self, server_round, loss, metrics):
        """log round's centralized metrics"""
        self._record("centralized", server_round, loss, metrics)

    def log_distributed(self, server_round, loss, metrics):
        """log round's distributed metrics"""
        self._record("distributed", server_round, loss, metrics)

    def _record(self, split, server_round, loss, metrics):
        r = server_round + self.round_offset
        entry = self._data[split].setdefault(r, {})
        if loss is not None:
            entry["loss"] = float(loss)
        for k, v in (metrics or {}).items():
            entry[k] = float(v)
        self._flush()

    # persistence and helpers
    def _flush(self):
        self._write_json()
        self._write_csv()

    def _write_json(self):
        out = {}
        for split, rounds in self._data.items():
            metric_series = {}
            for rnd in sorted(rounds.keys()):
                for metric, value in rounds[rnd].items():
                    metric_series.setdefault(metric, []).append([rnd, value])
            out[split] = metric_series
        with open(self.metrics_json_path, "w") as f:
            json.dump(out, f, indent=2)

    def _write_csv(self):
        columns = []
        for split in ("train", "centralized", "distributed"):
            metrics = set()
            for entry in self._data[split].values():
                metrics.update(entry.keys())
            for metric in sorted(metrics):
                columns.append((split, metric))

        all_rounds = sorted({r for rounds in self._data.values() for r in rounds})

        with open(self.metrics_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round"] + [f"{split}_{metric}" for split, metric in columns])
            for rnd in all_rounds:
                row = [rnd]
                for split, metric in columns:
                    row.append(self._data[split].get(rnd, {}).get(metric, ""))
                writer.writerow(row)

def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)

class _RecordingStrategyMixin:
    """use flwr's FedAvg / FedAdam, but also record metrics"""

    def attach_recorder(self, recorder):
        self._recorder = recorder
        return self

    def aggregate_fit(self, server_round, results, failures):

        # aggregate like flwr does
        aggregated = super().aggregate_fit(server_round, results, failures)

        # record extra metrics
        recorder = getattr(self, "_recorder", None)
        if recorder is not None and results:
            total = sum(fit_res.num_examples for _, fit_res in results)
            if total > 0:
                train_loss = sum(
                    fit_res.num_examples * float(fit_res.metrics.get("loss", 0.0))
                    for _, fit_res in results
                ) / total
                recorder.log_train_loss(server_round, train_loss)

        # save checkpoint
        if recorder is not None and aggregated is not None:
            params, _ = aggregated
            abs_round = server_round + recorder.round_offset
            if params is not None and abs_round % recorder.checkpoint_every == 0:
                adam_state = None
                if getattr(self, "m_t", None) is not None and getattr(self, "v_t", None) is not None:
                    adam_state = (self.m_t, self.v_t)
                recorder.save_latest(abs_round, parameters_to_ndarrays(params), adam_state)
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):

        # aggregate like flwr does
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        # record extra metrics
        recorder = getattr(self, "_recorder", None)
        if recorder is not None and aggregated is not None:
            loss, metrics = aggregated
            if loss is not None:
                recorder.log_distributed(server_round, loss, metrics)
        return aggregated

class RecordingFedAvg(_RecordingStrategyMixin, FedAvg):
    pass

class RecordingFedAdam(_RecordingStrategyMixin, FedAdam):
    pass
