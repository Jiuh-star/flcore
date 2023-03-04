from __future__ import annotations

import dataclasses
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Iterator, TypeVar

import rich.progress
import rich.repr
import rich.table
import rich.text
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .client import ClientProtocol
from .server import Server, EvaluationResult
from .utils import model as model_utils

# TODO: straightforward replay & low_memory

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


@dataclasses.dataclass
class Replay:
    logbook: list[LogItem]
    clients: list[ClientProtocol]
    model: nn.Module


class FederatedLearning(ABC):
    def __init__(self, *, server: Server, log_dir: str | pathlib.Path, low_memory: bool = False,
                 tensorboard: bool = True):
        self.server = server
        self.log_dir = pathlib.Path(log_dir)
        self.tensorboard = tb.SummaryWriter(log_dir) if tensorboard else None
        self.logbook: list[LogItem] = []
        self.progress = _Progress()

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if low_memory:
            self._set_system_low_memory()

    @abstractmethod
    def algorithm(self) -> nn.Module:
        ...

    def run(self):
        if len(self.server.registered_clients) == 0:
            raise ValueError("There are no registered clients in server.")

        with self.progress:
            model = self.algorithm()

        return model

    def log(self, log_item: LogItem, *, big_item: bool = False, filename: str = None):
        self.progress.log(log_item)
        self.logbook.append(log_item)

        # Save big log_item into an independent file.
        if big_item:
            self.logbook.pop()
            filename = filename or f"log-{log_item.epoch}-{time.strftime('%X')}.pth"
            torch.save(log_item, self.log_dir / filename)

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
        torch.save(self.logbook, self.log_dir / "logbook.pth")

    def save_replay(self, replay: Replay, filename: str | pathlib.Path = None):
        torch.save(replay, self.log_dir / (filename or "replay.pth"))

    def load_replay(self, filename: str | pathlib.Path = None):
        replay: Replay = torch.load(self.log_dir / (filename or "replay.pth"))

        last_epoch = replay.logbook[-1].epoch
        num_client = len(replay.clients)

        if num_client != len(self.server.registered_clients):
            raise ValueError("The number of clients in replay file does not equals to "
                             "the number of registered clients in server.")

        if not model_utils.compare_model_structures(replay.model, self.server.model,
                                                    *[client.model for client in replay.clients],
                                                    *[client.model for client in self.server.registered_clients],
                                                    raise_error=False):
            raise ValueError("The model structure in replay file dose not equals to "
                             "the structure of registered clients in server.")

        # There are some other exception to be handled, local epoch equality check etc., but that will be much more
        # complex. Therefore, we assert scholar would load replay of the same system saved replay. Also, SEED was set.

        self.progress.log("Replay Warning: The system is going to replay, make sure that the seed was set.",
                          style="red bold")
        self.progress.log(f"Replay: {num_client} clients will replay to global epoch {last_epoch}.")
        self._set_system_replay(replay)

    def _set_system_replay(self, replay: Replay):
        """
        Dynamic substitute method ``server.evaluate()`` to skip real evaluation and return evaluation result in
        `replay`. And substitute ``client.train()`` to do nothing function. Clean these patch when replay done.
        """
        metrics = [log_item for log_item in replay.logbook if log_item.metrics]
        metrics_iter = iter(metrics)
        original_evaluate = self.server.evaluate

        # replayed server evaluate
        def replayed_evaluate() -> EvaluationResult:
            for log_item in metrics_iter:
                if log_item == metrics[-1]:
                    replay_done()

                return log_item.metrics

        def replay_done():
            # Server replay done: clean patch in server
            self.server.evaluate = original_evaluate
            self.server.model = replay.model

            # Client replay done: Drop old clients and load replay clients
            if len(self.server.registered_clients) != len(replay.clients):
                raise warnings.warn(f"The number of registered clients ({len(self.server.registered_clients)}) "
                                    f"not equals to the number of clients ({len(replay.clients)}) in replay file, "
                                    f"which may caused by dynamic register when system running.")

            self.server.registered_clients = replay.clients

        # Client replay
        for client in self.server.registered_clients:
            setattr(client, "_original_train_", client.train)
            client.train = lambda dataloader: None

        # Server replay
        self.server.evaluate = replayed_evaluate

    def _set_system_low_memory(self):
        """
        Set client model and optimizer to ``None`` when registering. Remove loaded model and optimizer at last epoch
        and load model and optimizer when ``server.select_clients()`` was called.
        """
        original_register_client = self.server.register_client
        original_unregister_client = self.server.unregister_client
        original_select_clients = self.server.select_clients

        low_memory_dir = self.log_dir / "low_memory"
        low_memory_dir.mkdir(parents=True, exist_ok=True)

        def low_memory_register_client(client: ClientProtocol):
            torch.save([client.model, client.optimizer], low_memory_dir / f"{client.id}.lm")
            client.model, client.optimizer = None, None
            original_register_client(client)

        def low_memory_unregister_client(id_: str) -> ClientProtocol | None:
            if client := original_unregister_client(id_):
                client.model, client.optimizer = torch.load(low_memory_dir / f"{client.id}.lm")
            return client

        def low_memory_select_clients() -> list[ClientProtocol]:
            selected_clients = original_select_clients()
            for client in selected_clients:
                client.model, client.optimizer = torch.load(low_memory_dir / f"{client.id}.lm")

            return selected_clients

        self.server.register_client = low_memory_register_client
        self.server.unregister_client = low_memory_unregister_client
        self.server.select_clients = low_memory_select_clients


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
