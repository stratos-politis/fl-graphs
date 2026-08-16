from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters, NDArrays, Scalar

import os

import torch
from torch_geometric.loader import DataLoader

from typing import Dict, Tuple, Optional
from collections import OrderedDict

from dataset import WyzeGraphDataset, prepare_holdout_dataloader
from model import build_model, version_from_flags
from task import eval_client
from results_logger import setup_logger, ResultsRecorder, RecordingFedAvg, RecordingFedAdam

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = setup_logger(__name__, "[SERVER]")

def get_on_fit_config(run_config: dict):
    """Dictates client behaviour (learning rate, optimizer etc)."""

    base_lr = run_config["lr"]

    def fit_config(server_round: int):

        # Step decay (config via pyproject, set to 1.0 for no decay)
        decay_factor = run_config.get("lr-decay-factor", 0.85)
        drop_every_n_rounds = run_config.get("lr-decay-every", 2)

        current_lr = base_lr * (decay_factor ** (server_round // drop_every_n_rounds))

        if server_round == 1:
            log.info(f"Round {server_round}: Initializing fit config to clients (LR: {current_lr:.6f})")
        else:
            log.info(f"Round {server_round}: Updating fit config to clients (LR: {current_lr:.6f})")

        return {
            "lr": current_lr,
            "local_epochs": run_config["local-epochs"],
            "optimizer": run_config.get("optimizer", "adam"),
            "weight_decay": run_config.get("weight-decay", 1e-5),
            "loss_fn": str(run_config.get("loss-fn", "bce")),
            "num_negatives": int(run_config.get("num-negatives", 1)),
            "hard_negative_frac": float(run_config.get("hard-negative-frac", 0.0)),
        }
    return fit_config

def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Aggregates client metrics."""

    # Weight each client by its example count
    total_examples = sum([num_examples for num_examples, _ in metrics])

    aggregated_metrics = {}
    for key in metrics[0][1].keys():
        aggregated_metrics[key] = sum([num_examples * m[key] for num_examples, m in metrics]) / total_examples

    return aggregated_metrics

def get_evaluate_fn(val_loader, device, num_node_features, num_rule_types, num_trigger_states, num_action_types, recorder=None, version="v2"):
    """Centralized eval on server's global model instead of each client's local eval."""

    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        model = build_model(
            version,
            num_node_features=num_node_features,
            num_rule_types=num_rule_types,
            num_trigger_states=num_trigger_states,
            num_action_types=num_action_types,
        ).to(device)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, metrics, _ = eval_client(model, val_loader, device, verbose=False)

        if recorder is not None:
            recorder.log_centralized(server_round, loss, metrics)

            # Select best model by MRR, not loss
            saved = recorder.maybe_save_best(
                server_round, metrics.get("mrr"), model.state_dict(),
                metric_name="mrr", higher_is_better=True
            )
            if saved:
                log.info(f"Round {server_round}: new best MRR={metrics.get('mrr'):.4f} "
                         f"→ saved {recorder.best_model_path}")

        return loss, metrics

    return evaluate

def server_fn(context: Context):
    """Generates and initializes Flower server."""

    # 1. Define server device
    device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

    # Retrieve config from pyproject.toml
    num_rounds = context.run_config["num-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    min_available_clients = context.run_config["min-available-clients"]
    min_fit_clients = context.run_config["min-fit-clients"]
    min_evaluate_clients = context.run_config["min-evaluate-clients"]
    batch_size = context.run_config["batch-size"]

    # Checkpoints to resume broken/paused runs
    resume_from = str(context.run_config.get("resume-from", "") or "")
    checkpoint_every = int(context.run_config.get("checkpoint-every", 1) or 1)
    resume_ckpt = None
    round_offset = 0
    if resume_from:
        ckpt_path = resume_from
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "latest_checkpoint.pt")
        resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        round_offset = int(resume_ckpt["round"])
        resume_dir = os.path.dirname(ckpt_path)
        log.info(f"Resuming run '{resume_dir}' from round {round_offset} "
                 f"(target total: {num_rounds} rounds)")

    # Instantiate dataset to build caches, vocabulary and dimensions
    log.info(f"Initializing dataset to extract global vocabulary sizes and build caches...")
    train_dataset = WyzeGraphDataset(context=context, mode='train')

    # Pre-cache test dataset
    log.info(f"Pre-caching test dataset...")
    test_dataset = WyzeGraphDataset(
        context=context,
        mode='test',
        train_rule2id=train_dataset.rule2id,
        train_device2id=train_dataset.device2id
    )

    # Eval set:
    # - test set (cold start) -> default, full LOO
    # - IID slice of TRAIN set -> 2nd place eval (train-holdout-frac>0)

    holdout_frac = context.run_config.get("train-holdout-frac", 0.0)
    if holdout_frac > 0:
        global_val_loader = prepare_holdout_dataloader(train_dataset, holdout_frac, batch_size=batch_size)
        log.info(f"Evaluating on held-out TRAIN slice (frac={holdout_frac}): "
                 f"{len(global_val_loader.dataset)} users")
    else:
        global_val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_node_features = train_dataset.num_inputs
    num_rule_types = train_dataset.num_rule_types
    log.info(f"Vocabulary built. Nodes: {num_node_features}, Rules: {num_rule_types}")

    # Set up results recording
    strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()
    if resume_ckpt is not None:
        recorder = ResultsRecorder(
            resume_dir=resume_dir,
            round_offset=round_offset,
            checkpoint_every=checkpoint_every,
        )
    else:
        recorder = ResultsRecorder(
            base_dir="results",
            run_config=context.run_config,
            extra_meta={
                "strategy": strategy_name,
                "device": str(device),
                "num_node_features": num_node_features,
                "num_rule_types": num_rule_types,
            },
            checkpoint_every=checkpoint_every,
        )
    log.info(f"Recording results to: {recorder.run_dir}")

    # Configure model versions
    model_type = str(context.run_config.get("model", "edge"))
    decoder = str(context.run_config.get("decoder", "mlp"))
    encoder = str(context.run_config.get("encoder", "sage"))
    version = version_from_flags(model_type, encoder, decoder)
    global_model = build_model(
        version,
        num_node_features=num_node_features,
        num_rule_types=num_rule_types,
        num_trigger_states=train_dataset.num_trigger_states,
        num_action_types=train_dataset.num_action_types,
    )

    # Initial (round 0) parameters
    if resume_ckpt is not None:
        initial_parameters = ndarrays_to_parameters(resume_ckpt["params"])
    else:
        initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        )

    # Configure strategy
    common_kwargs = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        on_fit_config_fn=get_on_fit_config(context.run_config),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(
            val_loader=global_val_loader,
            device=device,
            num_node_features=num_node_features,
            num_rule_types=num_rule_types,
            num_trigger_states=train_dataset.num_trigger_states,
            num_action_types=train_dataset.num_action_types,
            recorder=recorder,
            version=version,
        ),
    )
    if strategy_name == "fedadam":
        strategy = RecordingFedAdam(**common_kwargs, eta=1e-2, beta_1=0.9, beta_2=0.99, tau=1e-3)
    else:
        strategy = RecordingFedAvg(**common_kwargs)
    strategy.attach_recorder(recorder)
    log.info(f"Strategy: {strategy_name}")

    # Restore FedAdam server optimizer moments so a resumed run continues exactly (only if FedAdam)
    if resume_ckpt is not None and resume_ckpt.get("m_t") is not None:
        strategy.m_t = resume_ckpt["m_t"]
        strategy.v_t = resume_ckpt["v_t"]
        log.info("Restored FedAdam optimizer state (m_t, v_t) from checkpoint")

    # Run only remaining rounds when resuming
    remaining_rounds = num_rounds - round_offset if resume_ckpt is not None else num_rounds
    if remaining_rounds <= 0:
        log.info(f"Target of {num_rounds} rounds already reached at round {round_offset}; nothing to do.")
    config = ServerConfig(num_rounds=max(0, remaining_rounds))

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)