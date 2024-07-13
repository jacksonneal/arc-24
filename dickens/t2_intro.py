# ruff: noqa: S108
"""Classifier for continuous XOR dataset."""
from __future__ import annotations

import inspect
import logging
import sys
from typing import TYPE_CHECKING, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import loguru
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint
from flax.training import train_state
from loguru import logger
from matplotlib.colors import to_rgba
from torch.utils import data
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy._typing import NDArray


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def my_filter(record: loguru.Record) -> bool:
    if record["name"] in {
        "absl.logging",
        "asyncio.selector_events",
    } or record["name"].startswith("jax._src"):
        return False
    return True


logger.remove()
logger.add(sys.stderr, filter=my_filter, level="INFO")


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

    labels: NDArray[np.int32]
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
        labels = (data.sum(axis=1) == 1).astype(np.int32)
        # add noise
        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.labels = labels

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
    epoch_cb: Callable[[int, train_state.TrainState], None],
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
        epoch_cb:
            called each epoch with current train state

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
        epoch_cb(epoch, state)
    return state


def visualize_samples(data: NDArray[np.float32], labels: NDArray[np.int32]) -> None:
    """Plot XOR data and labels."""
    data_0 = data[labels == 0]
    data_1 = data[labels == 1]

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

    train_dataset = XORDataset(size=2500, seed=42, std=0.1)
    train_data_loader = data.DataLoader(
        train_dataset,
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

    # visualize_samples(train_dataset.data, train_dataset.labels)
    # plt.show()

    with orbax.checkpoint.CheckpointManager(
        orbax.checkpoint.test_utils.erase_and_create_empty(
            "/tmp/flax_ckpt/orbax/managed"
        ),
        options=orbax.checkpoint.CheckpointManagerOptions(),
    ) as ckpt_mngr:

        def epoch_cb(epoch, state) -> None:
            ckpt_mngr.save(epoch, args=orbax.checkpoint.args.StandardSave(state))
            ckpt_mngr.wait_until_finished()

        trained_state = train_model(
            model_state, train_data_loader, epoch_cb, num_epochs=100
        )

        restored_state = ckpt_mngr.restore(
            ckpt_mngr.latest_step(),
            args=orbax.checkpoint.args.StandardRestore(trained_state),
        )

        test_dataset = XORDataset(size=500, seed=123, std=0.1)
        test_data_loader = data.DataLoader(
            test_dataset, batch_size=128, collate_fn=numpy_collate
        )

        def eval_model(
            state: train_state.TrainState, data_loader: data.DataLoader
        ) -> None:
            all_accs, batch_sizes = [], []
            for batch in data_loader:
                batch_acc = eval_step(state, batch)
                all_accs.append(batch_acc)
                batch_sizes.append(batch[0].shape[0])
            # Weighted average since some batches might be smaller
            acc = sum(
                [a * b for a, b in zip(all_accs, batch_sizes, strict=True)]
            ) / sum(batch_sizes)
            logger.debug(f"Accuracy of the model: {100.0 * acc:4.2f}%")

        eval_model(restored_state, test_data_loader)

        def visualize_classification(model, data, labels):
            data_0 = data[labels == 0]
            data_1 = data[labels == 1]

            fig = plt.figure(figsize=(4, 4), dpi=500)
            plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
            plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
            plt.title("Dataset samples")
            plt.ylabel(r"$x_2$")
            plt.xlabel(r"$x_1$")
            plt.legend()

            # Let's make use of a lot of operations we have learned above
            c0 = np.array(to_rgba("C0"))
            c1 = np.array(to_rgba("C1"))
            x1 = jnp.arange(-0.5, 1.5, step=0.01)
            x2 = jnp.arange(-0.5, 1.5, step=0.01)
            xx1, xx2 = jnp.meshgrid(
                x1, x2, indexing="ij"
            )  # Meshgrid function as in numpy
            model_inputs = np.stack([xx1, xx2], axis=-1)
            logits = model(model_inputs)
            preds = nn.sigmoid(logits)
            output_image = (1 - preds) * c0[None, None] + preds * c1[
                None, None
            ]  # Specifying "None" in a dimension creates a new one
            output_image = jax.device_get(
                output_image
            )  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
            plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
            plt.grid(False)
            return fig

        _ = visualize_classification(
            model.bind(restored_state.params), test_dataset.data, test_dataset.labels
        )
        plt.show()
