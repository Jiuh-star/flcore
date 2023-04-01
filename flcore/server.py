from __future__ import annotations

import copy
import math
import random
from typing import Optional, Sequence, TypeVar, NewType, Callable

import torch
import torch.utils.data as data

from .client import MetricResult, ClientProtocol
from .utils import model as model_utils
from .utils import robust

Client = TypeVar("Client", bound=ClientProtocol)
EvaluationResult = NewType("EvaluationResult", dict[str, float | dict[str, float]])


class Server:
    def __init__(self, *, select_ratio: float, max_epoch: int, learning_rate: float = 1.0,
                 robust_fn: robust.RobustFn = None):
        """
        The server in federated learning. It works with some clients in ``FederatedLearning``.

        You can specify global learning rate in `learning_rate`. The global learning rate may distinctly slow down the
        convergence time but take smoother loss variance in return.

        :param select_ratio: Client select ratio of each round in federated learning.
        :param max_epoch: The max epoch of server, namely global rounds of federated learning.
        :param learning_rate: The global learning rate, this works in aggregation.
        :param robust_fn: The robust aggregation function.
        """
        self.select_ratio = select_ratio
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.registered_clients: list[Client | ClientProtocol] = []
        self.robust_fn = robust_fn
        self._model: Optional[torch.nn.Module] = None

    def register_client(self, client: Client):
        """
        Register a client to server.

        :param client: Client to be registered.

        :raises ValueError: When client is not an available client.
        :raises RuntimeError: When there are duplicated client id.
        """
        if not isinstance(client, ClientProtocol):
            raise ValueError(f"The client is not an available client.")

        if self.get_client(id_=client.id):
            raise RuntimeError(f'Conflicted client id "{client.id}"')

        self.registered_clients.append(client)

    def unregister_client(self, id_: str) -> Client | None:
        """
        Unregister a client in server, return this client.

        :param id_: Client id
        :return: A client or None if no this client
        """

        for i, client in enumerate(self.registered_clients):
            if client.id == id_:
                return self.registered_clients.pop()
        return None

    def get_client(self, id_: str) -> Client | None:
        """
        Get the registered client of `id`.

        :param id_: Client's id
        :return: The client, or None if not found
        """
        for client in self.registered_clients:
            if client.id == id_:
                return client

    def select_clients(self) -> list[Client]:
        """
        Randomly select ``int(select_ratio * num_registered_clients)`` clients.

        :return: Selected clients
        """
        num_select = int(len(self.registered_clients) * self.select_ratio)
        selected_clients = random.sample(self.registered_clients, k=num_select)
        return selected_clients

    @property
    def model(self) -> torch.nn.Module:
        """
        Server's model, namely global model.
        """
        if self._model is None:
            assert len(self.registered_clients) > 0

            self._model = copy.deepcopy(self.registered_clients[0].model)
            self._model.requires_grad_(False)
            # average model initiate parameters first will make the model non-trainable.
            # self.aggregate(self.registered_clients)

        return self._model

    @model.setter
    def model(self, new_model: torch.nn.Module):
        self._model = new_model.cpu()

    def aggregate(self, clients: Sequence[ClientProtocol], weights: Sequence[float] = None,
                  robust_fn: robust.RobustFn = None):
        """
        Aggregate local models to global model by aggregating updates (delta of models). Each model update will be
        multiplied by server's learning rate and corresponding weight to perform aggregation.

        :param clients: Clients to be aggregated.
        :param weights: Weight that corresponding model update, expected the sum equals to 1.
        :param robust_fn: Robust function, specified it explicitly to make the code in FederatedLearning
        straightforward.

        :raises ValueError: When no client in clients.
        :raises ValueError: When sum of weights is not 1.
        """
        if len(clients) == 0:
            raise ValueError("Not enough clients to perform aggregation.")

        if weights is None:
            dataset_size = sum([len(client.train_dataset) for client in clients])
            weights = [len(client.train_dataset) / dataset_size for client in clients]

        if len(weights) != len(clients):
            raise ValueError(f"The length of weights ({len(weights)}) and clients ({len(clients)}) are not the same.")

        if not math.isclose(math.fsum(weights), 1., abs_tol=1E-5):
            raise ValueError(f"The sum of weights should close to 1, got {sum(weights)}.")

        local_models = [client.model for client in clients]

        if robust_fn := robust_fn or self.robust_fn:
            local_models, weights = robust_fn(self.model, local_models, weights)
            weights = [1 / len(local_models)] * len(local_models) if weights is None else weights

        self.model = model_utils.aggregate_model(self.model, local_models, weights)

    def evaluate(self) -> EvaluationResult:
        """
        Evaluate global model on all registered clients and compute mean and std for each metric.

        :return: Mean, std and raw values of client evaluation result of each metric.
        """
        return self._eval(lambda client: client.eval_dataloader)

    def test(self) -> EvaluationResult:
        """
        Test global model on all registered clients and compute mean and std for each metric.

        :return: Mean, std and raw values of client test result of each metric.
        """
        return self._eval(lambda client: client.test_dataloader)

    def _eval(self, load_dataloader: Callable[[ClientProtocol], data.DataLoader]) -> EvaluationResult:
        """ evaluate all models in client with dataloader from load_dataloader. """
        self.registered_clients: list[ClientProtocol]
        client_metric_result: dict[str, MetricResult]

        for client in self.registered_clients:
            client.receive_model(self.model)

        client_metric_result = {
            client.id: client.evaluate(load_dataloader(client))
            for client in self.registered_clients
        }

        metric_client_result = self._collect_evaluation_results(client_metric_result)
        evaluation_result = self._analyze_evaluation_results(metric_client_result)

        return EvaluationResult(evaluation_result)

    @staticmethod
    def _collect_evaluation_results(client_metric_result: dict[str, MetricResult]) -> dict[str, dict[str, float]]:
        """
        Rotate a recursive dict ``{client: {metric, result}}`` to ``{metric: {client: result}}``.
        """
        if not client_metric_result:  # nothing to rotate
            return {}

        sample = list(client_metric_result.values())[0]  # a MetricResult instance
        metrics = list(sample.keys())  # get metric names

        # do rotate
        metric_client_result = {}
        for metric in metrics:
            metric_client_result[metric] = {client: mr[metric] for client, mr in client_metric_result.items()}

        return metric_client_result

    @staticmethod
    def _analyze_evaluation_results(metric_client_result: dict[str, dict[str, float]]) -> EvaluationResult:
        """
        Compute mean and std of each metric results.
        """
        evaluation_result = {}
        for metric, client_result in metric_client_result.items():
            results = list(client_result.values())
            evaluation_result[f"{metric} (mean)"] = torch.tensor(results).mean().item()
            evaluation_result[f"{metric} (std)"] = torch.tensor(results).std(None).item()
            evaluation_result[f"{metric} (raw)"] = client_result
        return EvaluationResult(evaluation_result)
