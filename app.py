# ruff: noqa: D101
from pathlib import Path
from typing import Annotated

import msgspec
from loguru import logger

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

with Path.open(training_challenge_json_path, "rb") as f:
    data = msgspec.json.decode(f.read(), type=TaskDict)

    for task_id, task in data.items():
        logger.debug(
            f"task ID: {task_id},"
            f" num test: {len(task.test)},"
            f" num train: {len(task.train)}",
        )
