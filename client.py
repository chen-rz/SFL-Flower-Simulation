from collections import OrderedDict
from pathlib import Path
import random
from typing import Dict, List

import flwr as fl
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader
import torchvision
from flwr.common import Scalar

from dataset_utils import get_dataloader
import utils
from constants import MODEL_TYPE, EPOCHS, BATCH_SIZE
from model_statistics import model_statistics


# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, param_dict: dict):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # TODO Instantiate more models here
        if MODEL_TYPE == "alexnet":
            self.net = torchvision.models.alexnet(num_classes=10)
        elif MODEL_TYPE == "vgg11":
            self.net = torchvision.models.vgg11(num_classes=10)
        else:
            raise ValueError("Model type not supported")

        # TODO Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Randomly initialized parameters
        self.properties["dataSize"] = param_dict["dataSize"]
        self.properties["computation"] = param_dict["computation"] * random.uniform(0.75, 1.25)
        self.properties["transPower"] = param_dict["transPower"] * random.uniform(0.75, 1.25)
        self.properties["channelGain"] = param_dict["channelGain"] * random.uniform(0.75, 1.25)
        self.properties["splitLayer"] = param_dict["splitLayer"]

    def get_properties(self, config) -> Dict[str, Scalar]:
        return self.properties

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        # Set model weights
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(
            ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train_loss, mean_square_batch_loss = utils.train(
            self.net, trainloader, epochs=config["epochs"], device=self.device
        )

        # Record loss
        with open(
                "./output/train_loss/client_{}.txt".format(self.cid),
                mode='a'
        ) as outputFile:
            outputFile.write(str(train_loss) + "\n")
        
        with open(
                "./output/mean_square_batch_loss/client_{}.txt".format(self.cid),
                mode='a'
        ) as outputFile:
            outputFile.write(str(mean_square_batch_loss) + "\n")

        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(
            ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = utils.test(self.net, valloader, device=self.device)

        # Record loss and accuracy
        with open(
                "./output/val_loss/client_{}.txt".format(self.cid),
                mode='a'
        ) as outputFile:
            outputFile.write(str(loss) + "\n")
        with open(
                "./output/val_accu/client_{}.txt".format(self.cid),
                mode='a'
        ) as outputFile:
            outputFile.write(str(accuracy) + "\n")

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": EPOCHS,  # number of local epochs
        "batch_size": BATCH_SIZE,
    }
    return config


def get_params(model) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v))
                             for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(test_set: torchvision.datasets.CIFAR10, ):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        # TODO Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO Instantiate more models here
        if MODEL_TYPE == "alexnet":
            model = torchvision.models.alexnet(num_classes=10)
        elif MODEL_TYPE == "vgg11":
            model = torchvision.models.vgg11(num_classes=10)
        else:
            raise ValueError("Model type not supported")

        set_params(model, parameters)
        model.to(device)

        testloader = DataLoader(test_set, batch_size=50, num_workers=25)
        loss, accuracy = utils.test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
