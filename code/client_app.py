import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # pin sim to a single gpu

import torch

from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from dataset import WyzeGraphDataset, prepare_client_dataloader
from model import build_model, version_from_flags
from task import train_client, eval_client
from results_logger import setup_logger

import gc

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
log = setup_logger(__name__, "[CLIENT]")

# Load ONCE and share across all simulated clients
GLOBAL_TRAIN_DATASET = None
GLOBAL_TEST_DATASET = None

class WyzeClient(NumPyClient):
    def __init__(self, cid, model, trainloader, valloader, lr, local_epochs, device):

        self.cid = cid
        self.should_log = (cid == "0")
        self.device = device

        self.model = model
        self.local_epochs = local_epochs
        self.lr = lr

        self.trainloader = trainloader
        self.valloader = valloader

        if self.should_log:
            log.info(f"[Client {cid}] Loaded train dataset: {len(self.trainloader.dataset)} samples")
            log.info(f"[Client {cid}] Loaded validation dataset: {len(self.valloader.dataset)} samples")
            log.info(f"[Client {cid}] Loaded training model")
            log.info(f"[Client {cid}] Tried accelerators: CUDA -> MPS -> CPU. Using device: {self.device}")

    def set_parameters(self, parameters):
        """Copies model parameters sent by server to client's local model"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        if self.should_log:
            log.info(f"[Client {self.cid}] Fit: Received server's updated parameters")

    def get_parameters(self, config):
        """Sends model parameters back to the server."""
        if self.should_log:
            log.info(f"[Client {self.cid}] Fit: Sent local parameters to server")

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Performs one round of local training."""
        if self.should_log:
            log.info(f"[Client {self.cid}] Fit: Started local training")

        # Update local model with global parameters
        self.set_parameters(parameters)

        # Get learning parameters
        lr = float(config["lr"])
        epochs = int(config["local_epochs"])

        if self.should_log:
            log.info(f"[Client {self.cid}] Fit: Received learning parameters: lr={lr:.6f}, epochs={epochs}")

        # Get optimizer from server
        opt_name = str(config.get("optimizer", "adam")).lower()
        weight_decay = float(config.get("weight_decay", 1e-5))
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.should_log:
            log.info(f"[Client {self.cid}] Fit: Configured optimizer: {opt_name}")

        # Get selected loss from server
        loss_fn = str(config.get("loss_fn", "bce"))
        num_negatives = int(config.get("num_negatives", 1))
        hard_frac = float(config.get("hard_negative_frac", 0.0))

        # Perform local training
        loss = train_client(
            model=self.model,
            trainloader=self.trainloader,
            epochs=epochs,
            optimizer=optimizer,
            device=self.device,
            verbose=self.should_log,
            diagnostic_mode=None,
            loss_fn=loss_fn,
            num_negatives=num_negatives,
            hard_frac=hard_frac,
        )

        if self.should_log:
            log.info(f"[Client {self.cid}] Fit: Training completed")

        # Delete garbage
        del optimizer
        gc.collect()

        # Return updated params, local sample count (FedAvg's aggregation weight), and loss
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": loss}

    def evaluate(self, parameters, config):
        """Evaluates state of global model on local data without updating it."""

        # Update local model parameters
        self.set_parameters(parameters)

        # Perform testing on validation data
        loss, metrics, num_eval = eval_client(self.model, self.valloader, self.device, verbose=self.should_log)
        if self.should_log:
            log.info(f"[Client {self.cid}] Eval: Loss: {loss:.4f} | Metrics: {metrics}")

        # Delete garbage
        gc.collect()

        return float(loss), num_eval, metrics

def client_fn(context: Context):
    """Generates and initializes Flower clients with unique Client IDs."""

    # Declare globals
    global GLOBAL_TRAIN_DATASET, GLOBAL_TEST_DATASET

    # Data partitioning parameters
    batch_size = context.run_config["batch-size"]
    num_partitions = context.node_config["num-partitions"]

    # Model & training parameters
    local_epochs = context.run_config["local-epochs"]
    lr = context.run_config["lr"]
    holdout_frac = context.run_config.get("train-holdout-frac", 0.0)

    # Unique ID for each client
    partition_id = context.node_config["partition-id"]

    # Set up device
    device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

    # Initialize dataset
    if GLOBAL_TRAIN_DATASET is None:
        log.info(f"Loading global TRAIN dataset into RAM for all clients...")
        GLOBAL_TRAIN_DATASET = WyzeGraphDataset(context=context, mode='train')

    if GLOBAL_TEST_DATASET is None:
        log.info(f"Loading global TEST dataset into RAM for all clients...")
        GLOBAL_TEST_DATASET = WyzeGraphDataset(
            context=context,
            mode='test',
            train_rule2id=GLOBAL_TRAIN_DATASET.rule2id,
            train_device2id=GLOBAL_TRAIN_DATASET.device2id
        )

    train_dataset = GLOBAL_TRAIN_DATASET
    test_dataset = GLOBAL_TEST_DATASET

    # Partition data for client
    trainloader = prepare_client_dataloader(
        dataset=train_dataset,
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        shuffle=True,
        holdout_frac=holdout_frac
    )

    valloader = prepare_client_dataloader(
        dataset=test_dataset,
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        shuffle=False
    )

    # Instantiate model version
    version = version_from_flags(
        context.run_config.get("model", "edge"),
        context.run_config.get("encoder", "sage"),
        context.run_config.get("decoder", "mlp"))
    model = build_model(
        version,
        num_node_features=train_dataset.num_inputs,
        num_rule_types=train_dataset.num_rule_types,
        num_trigger_states=train_dataset.num_trigger_states,
        num_action_types=train_dataset.num_action_types,
    ).to(device)

    # Return client object
    return WyzeClient(
        cid=str(partition_id),
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        lr=lr,
        local_epochs=local_epochs
    ).to_client()

app = ClientApp(client_fn=client_fn)