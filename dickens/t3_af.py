# ruff: noqa: D101, D102, D103, D107, PLR6301
"""Activation functions."""
import itertools
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
from loguru import logger
from torch import Tensor, nn

DATASET_PATH = Path("./data")
CHECKPOINT_PATH = Path("./saved_models/tutorial3")


def set_seed(seed: int) -> None:
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ActivationFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.config: dict[str, str | float] = {"name": self.name}


class Sigmoid(ActivationFunction):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))


class Tanh(ActivationFunction):
    def forward(self, x: Tensor) -> Tensor:
        x_exp, neg_x_exp = torch.exp(x), torch.exp(-x)
        return (x_exp - neg_x_exp) / (x_exp + neg_x_exp)


class ReLU(ActivationFunction):
    def forward(self, x: Tensor) -> Tensor:
        return x * (x > 0).float()


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.config["alpha"] = alpha

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x > 0, x, cast(float, self.config["alpha"]) * x)


class ELU(ActivationFunction):
    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x > 0, x, torch.exp(x) - 1)


class Swish(ActivationFunction):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


def get_grads(act_fn: ActivationFunction, x: Tensor) -> Tensor:
    """Computes the gradients of an activation function at specified positions.

    Inputs:
        act_fn - An object of the class "ActivationFunction" with an implemented forward pass.
        x - 1D input tensor.
    Output:
        A tensor with the same size of x containing the gradients of act_fn at x.
    """
    x = (
        x.clone().requires_grad_()
    )  # Mark the input as tensor for which we want to store gradients
    out = act_fn(x)
    out.sum().backward()  # Summing results in an equal gradient flow to each element in x
    assert x.grad is not None
    return x.grad  # Accessing the gradients of x by "x.grad"


def vis_act_fn(act_fn: ActivationFunction, ax: Any, x: Tensor) -> None:
    # Run activation function
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)
    # Push x, y and gradients back to cpu for plotting
    x, y, y_grads = x.cpu().numpy(), y.cpu().numpy(), y_grads.cpu().numpy()
    # Plotting
    ax.plot(x, y, linewidth=2, label="ActFn")
    ax.plot(x, y_grads, linewidth=2, label="Gradient")
    ax.set_title(act_fn.name)
    ax.legend()
    ax.set_ylim(-1.5, x.max())


act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish,
}


class BaseNetwork(nn.Module):
    def __init__(
        self,
        act_fn: ActivationFunction,
        input_size: int = 784,
        num_classes: int = 10,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        """Linear network with activation functions.

        Inputs:
            act_fn - Object of the activation function that should be used as non-linearity in the network.
            input_size - Size of the input images in pixels
            num_classes - Number of classes we want to predict
            hidden_sizes - A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 256, 128]

        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size, *hidden_sizes]
        for a, b in itertools.pairwise(layer_sizes):
            layers += [nn.Linear(a, b), act_fn]
        layers += [nn.Linear(layer_sizes[-1], num_classes)]
        self.layers = nn.Sequential(
            *layers
        )  # nn.Sequential summarizes a list of modules into a single module, applying them in sequence

        # We store all hyperparameters in a dictionary for saving and loading of the model
        self.config = {
            "act_fn": act_fn.config,
            "input_size": input_size,
            "num_classes": num_classes,
            "hidden_sizes": hidden_sizes,
        }

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)  # Reshape images to a flat vector
        out = self.layers(x)
        return out


def _get_config_file_path(model_path: Path, model_name: str) -> Path:
    # Name of the file for storing hyperparameter details
    return model_path / Path(model_name).with_suffix(".config")


def _get_model_file_path(model_path: Path, model_name: str) -> Path:
    # Name of the file for storing network parameters
    return model_path / Path(model_name).with_suffix(".tar")


def load_model(
    model_path: Path, model_name: str, net: BaseNetwork | None = None
) -> BaseNetwork:
    """Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        net - (Optional) If given, the state dict is loaded into this model. Otherwise, a new model is created.
    """
    config_file, model_file = (
        _get_config_file_path(model_path, model_name),
        _get_model_file_path(model_path, model_name),
    )
    assert Path.is_file(
        config_file
    ), f'Could not find the config file "{config_file}". Are you sure this is the correct path and you have your model config stored here?'
    assert Path.is_file(
        model_file
    ), f'Could not find the model file "{model_file}". Are you sure this is the correct path and you have your model stored here?'
    with Path.open(config_file, "r") as f:
        config_dict = json.load(f)
    if net is None:
        act_fn_name = config_dict["act_fn"].pop("name").lower()
        act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
        net = BaseNetwork(act_fn=act_fn, **config_dict)
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )
    net.load_state_dict(torch.load(model_file, map_location=device))
    return net


def save_model(model: BaseNetwork, model_path: Path, model_name: str) -> None:
    """Given a model, we save the state_dict and hyperparameters.

    Inputs:
        model - Network object to save parameters from
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = model.config
    Path.mkdir(model_path, exist_ok=True)
    config_file, model_file = (
        _get_config_file_path(model_path, model_name),
        _get_model_file_path(model_path, model_name),
    )
    with Path.open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    logger.debug("Running...")

    set_seed(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fetching the device that will be used throughout this notebook
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )
    logger.debug("Using device {device}", device=device)

    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial3/"
    # Files to download
    pretrained_files = [
        "FashionMNIST_elu.config",
        "FashionMNIST_elu.tar",
        "FashionMNIST_leakyrelu.config",
        "FashionMNIST_leakyrelu.tar",
        "FashionMNIST_relu.config",
        "FashionMNIST_relu.tar",
        "FashionMNIST_sigmoid.config",
        "FashionMNIST_sigmoid.tar",
        "FashionMNIST_swish.config",
        "FashionMNIST_swish.tar",
        "FashionMNIST_tanh.config",
        "FashionMNIST_tanh.tar",
    ]
    # Create checkpoint path if it doesn't exist yet
    Path.mkdir(CHECKPOINT_PATH, parents=True, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = CHECKPOINT_PATH / Path(file_name)
        if not Path.is_file(file_path):
            file_url = base_url + file_name
            logger.debug(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)  # noqa: S310
            except urllib.error.HTTPError:
                logger.exception(
                    "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n"
                )

    # Add activation functions if wanted
    act_fns = [act_fn() for act_fn in act_fn_by_name.values()]
    x = torch.linspace(
        -5, 5, 1000
    )  # Range on which we want to visualize the activation functions
    # Plotting
    rows = math.ceil(len(act_fns) / 2.0)
    fig, ax = plt.subplots(rows, 2, figsize=(8, rows * 4))
    for i, act_fn in enumerate(act_fns):
        vis_act_fn(act_fn, ax[divmod(i, 2)], x)
    fig.subplots_adjust(hspace=0.3)
    plt.show()
