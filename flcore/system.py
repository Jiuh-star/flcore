from __future__ import annotations

import dataclasses
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Sequence, Iterator, TypeVar

import rich.progress
import rich.repr
import rich.table
import rich.text
import torch.nn as nn
import torch.utils.tensorboard as tb

from .server import Server, EvaluationResult
from .utils import io

T = TypeVar("T")


@dataclasses.dataclass
class LogItem:
    epoch: int
    message: str = ""
    metrics: EvaluationResult = dataclasses.field(default_factory=dict)
    others: dict = dataclasses.field(default_factory=dict)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.epoch
        yield "message", self.message, ""
        # we only log float metrics in terminal to make it clear
        yield "metrics", {name: f"{value:.4f}" for name, value in self.metrics.items() if isinstance(value, float)}, {}
        yield "other", self.others, {}


class _SpeedColumn(rich.progress.ProgressColumn):
    def render(self, task: rich.progress.Task) -> rich.text.Text:
        """Show step-style speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return rich.text.Text("?", style="progress.data.speed")
        return rich.text.Text(f"{speed:.2f} step/s", style="progress.data.speed")


class _Progress(rich.progress.Progress):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("speed_estimate_period", 60 * 10)  # 10 minutes
        super().__init__(*args, **kwargs)
        self.columns = (
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(bar_width=None),
            _SpeedColumn(),
            rich.progress.MofNCompleteColumn(table_column=rich.table.Column(justify="center")),
            rich.progress.TimeElapsedColumn(),
            rich.progress.TimeRemainingColumn(),
        )
        self.header_task = {}

    def __call__(self, sequence: Sequence[T], header: str, **kwargs) -> Iterator[T]:
        """
        Track the `sequence` and print progress bar in terminal with title `header`.
        :param sequence: Sequence to be iterated.
        :param header: Title of progress bar.
        :param kwargs: Other keyword arguments in rich.progress.Progress.track()
        :return: The element in sequence.
        """
        if header not in self.header_task:
            self.header_task[header] = self.add_task(header)

        task_id = self.header_task[header]
        with self._lock:
            task = self.tasks[task_id]
            task.completed = 0
            task.finished_time = None
            task.finished_speed = None

        yield from super().track(sequence, task_id=task_id, **kwargs)


class FederatedLearning(ABC):
    def __init__(self, *, server: Server, log_dir: str | pathlib.Path, tensorboard: bool = True):
        self.server = server
        self.log_dir = pathlib.Path(log_dir)
        self.tensorboard = tb.SummaryWriter(log_dir) if tensorboard else None
        self.logbook: list[LogItem] = []
        self.progress = _Progress()

        self.log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def algorithm(self) -> nn.Module:
        ...

    def run(self):
        if len(self.server.registered_clients) == 0:
            raise ValueError("There are no registered clients in server.")

        try:
            with self.progress:
                model = self.algorithm()

        except KeyboardInterrupt:
            self.progress.log("Interruption detected, saving the system.", style="red bold")
            io.dump(self, self.log_dir / f"fl-{self.logbook[-1].epoch}.state", replace=True)

        return model

    def log(self, log_item: LogItem, *, big_item: bool = False, filename: str = None):
        self.progress.log(log_item)
        self.logbook.append(log_item)

        # Save big log_item into an independent file.
        if big_item:
            self.logbook.pop()
            filename = filename or f"log-{log_item.epoch}-{time.strftime('%X')}.pth"
            io.dump(log_item, self.log_dir / filename)

        # Plot metric in tensorboard
        if self.tensorboard:
            for metric, value in log_item.metrics.items():
                if isinstance(value, float):  # log scalar
                    self.tensorboard.add_scalar(metric, value, global_step=log_item.epoch)
                elif isinstance(value, dict):  # we don't log raw metric since its large quantity
                    pass

        # Save the new log_item into logbook.pth.
        # Note that torch.save doesn't support incremental save, we save the whole logbook instead,
        # so do not save any big log_item into logbook.
        io.dump(self.logbook, self.log_dir / "logbook.pth")
