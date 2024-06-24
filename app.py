# ruff: noqa: D101
from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import matplotlib.pyplot as plt
import msgspec
from loguru import logger
from matplotlib.colors import ListedColormap, Normalize

MIN_GRID_VAL = 0
MAX_GRID_VAL = 9

MIN_GRID_DIM = 1
MAX_GRID_DIM = 30

data_dir_path = Path("data")
json_ext_str = ".json"

training_challenge_json_path = data_dir_path / Path(
    "arc-agi_training_challenges.json",
).with_suffix(json_ext_str)


GridVal = Annotated[int, msgspec.Meta(ge=MIN_GRID_VAL, le=MAX_GRID_VAL)]


GridRow = Annotated[
    list[GridVal],
    msgspec.Meta(min_length=MIN_GRID_DIM, max_length=MAX_GRID_DIM),
]


Grid = Annotated[list[GridRow], msgspec]


class Test(msgspec.Struct):
    input_grid: Grid = msgspec.field(name="input")


class Train(msgspec.Struct):
    input_grid: Grid = msgspec.field(name="input")
    output_grid: Grid = msgspec.field(name="output")


class Task(msgspec.Struct):
    test: list[Test]
    train: list[Train]


TaskDict = dict[str, Task]


cmap = ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ],
)


def plot_task(task: Task) -> None:
    n_examples = len(task.train)
    norm = Normalize(vmin=0, vmax=9)
    _, axes = plt.subplots(2, n_examples, figsize=(n_examples * 4, 8))
    for column, example in enumerate(task.train):
        axes[0, column].imshow(example.input_grid, cmap=cmap, norm=norm)
        axes[1, column].imshow(example.output_grid, cmap=cmap, norm=norm)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    plt.show()


with Path.open(training_challenge_json_path, "rb") as f:
    task_dict = msgspec.json.decode(f.read(), type=TaskDict)

    logger.debug(f"num tasks: {len(task_dict)}")

    for task_id, task in task_dict.items():
        rot90 = True
        for train in task.train:
            input_arr = jnp.array(train.input_grid)
            output_arr = jnp.array(train.output_grid)

            if not jnp.array_equal(jnp.rot90(input_arr), output_arr):
                rot90 = False

        if rot90:
            logger.debug(
                "copy op"
                f" task ID: {task_id},"
                f" num test: {len(task.test)},"
                f" num train: {len(task.train)}",
            )
            plot_task(task)
