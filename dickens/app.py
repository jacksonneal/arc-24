"""Classifier for continuous XOR dataset."""

from collections.abc import Mapping
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training import train_state
from loguru import logger
from numpy._typing import NDArray
from torch.utils import data
from tqdm.auto import tqdm


class SimpleClassifier(nn.Module):
    """Dense classifier with a single hidden layer."""

    num_hidden: int
    """Number of hidden neurons."""

    num_outputs: int
    """Number of output neurons."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Perform forward pass."""
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x


class XORDataset(data.Dataset):
    """Continuous XOR dataset."""

    data: NDArray[np.float32]
    """XY pairs."""

    label: NDArray[np.int32]
    """Data labels."""

    def __init__(self, *, size: int, seed: int, std: float) -> None:
        """Initialize the dataset.

        Keyword Args
        ------------
            size:
                number of data points to generate
            seed:
                seed PRNG state for generating data points
            std:
                standard deviation of noise for continuous xor
        """
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed)
        self.std = std
        self._generate_continuous_xor()

    def _generate_continuous_xor(self) -> None:
        """Generate data and labels."""
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(
            np.float32,
        )
        label = (data.sum(axis=1) == 1).astype(np.int32)
        # add noise
        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.labels = label

    def __len__(self) -> int:
        """Number of data points."""
        return self.size

    def __getitem__(self, idx: int) -> tuple[NDArray[np.float32], np.int32]:
        """Access the ith data point.

        Args
        ----
            idx:
                index of data point to access

        Returns
        -------
            data point at index, tuple of data and labels
        """
        data_point = self.data[idx]
        data_label = self.labels[idx]
        return data_point, data_label


def numpy_collate(batch: Any) -> Any:
    """Merge a list of samples into a mini-batch."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple | list)):
        transposed = zip(*batch, strict=True)
        return [numpy_collate(samples) for samples in transposed]
    return np.array(batch)


def calculate_loss_acc(
    state: train_state.TrainState,
    params: Mapping,
    batch: list,
) -> tuple[jax.Array, jax.Array]:
    """Calculate loss and accuracy.

    Args
    ----
        state:
            model state
        params:
            model params
        batch:
            data and labels

    Returns
    -------
        tuple of loss and accuracy
    """
    data, labels = batch

    logits = state.apply_fn(params, data).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)

    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()

    return loss, acc


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: list,
) -> tuple[train_state.TrainState, jax.Array, jax.Array]:
    """Calculate loss for input, take gradients, update paramaters, return new state.

    Args
    ----
        state:
            model state
        batch:
            data and labels

    Returns
    -------
        tuple of new state, loss, and accuracy
    """
    val_grad_fn = jax.value_and_grad(
        calculate_loss_acc,
        argnums=1,  # params are second arg
        has_aux=True,  # additional outputs (accuracy)
    )
    (loss, acc), grads = val_grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


@jax.jit
def eval_step(
    state: train_state.TrainState,
    batch: list,
) -> jax.Array:
    """Determine model accuracy.

    Args
    ----
        state:
            model state
        batch:
            data and labels
    """
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc


def train_model(
    state: train_state.TrainState,
    data_loader: data.DataLoader,
    *,
    num_epochs: int,
) -> train_state.TrainState:
    """Execute training loop.

    Args
    ----
        state:
            model state
        data_loader:
            dataset and sampler

    Keyword Args
    ------------
        num_epochs:
            number of training loops to run

    Returns
    -------
        final model state
    """
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
            logger.debug(f"epoch: {epoch}, loss: {loss}, acc: {acc}")
    return state


def visualize_samples(data: NDArray[np.float32], label: NDArray[np.int32]) -> None:
    """Plot XOR data and labels."""
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()


if __name__ == "__main__":
    logger.debug("Running...")

    model = SimpleClassifier(num_hidden=8, num_outputs=1)
    rng = jax.random.PRNGKey(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (8, 2))  # batch size 8, input size 2
    params = model.init(init_rng, inp)

    dataset = XORDataset(size=2500, seed=42, std=0.1)
    data_loader = data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=numpy_collate,
    )

    optimizer = optax.sgd(learning_rate=0.1)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    trained_model_state = train_model(model_state, data_loader, num_epochs=100)

    # visualize_samples(dataset.data, dataset.label)
    # plt.show()

    # logger.debug(model.apply(params, inp))
