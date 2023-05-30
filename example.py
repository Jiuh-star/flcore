from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics as metrics
import torchvision as vision

import flcore
import flcore.utils.data as data_utils
from flcore.client import MetricResult


class FedAvgClient(flcore.ClientProtocol):
    def __init__(
            self,
            *,
            id_: str,
            max_epoch: int,
            learning_rate: float,
            train_dataloader: data.DataLoader,
            eval_dataloader: data.DataLoader,
            test_dataloader: data.DataLoader,
            device: torch.device,
            num_class: int,
    ):
        self.id = id_
        self.device = device
        self.dataset_size = len(train_dataloader.dataset)  # type: ignore
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics.MetricCollection(
            {
                "loss": metrics.MeanMetric(nan_strategy="error"),
                "accuracy": metrics.Accuracy(
                    "multiclass", num_classes=num_class, average="micro"
                ),
            }
        )

        self.metrics.to(self.device)

    def train(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.max_epoch):
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                y_hat = self.model(x)

                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()

    def _eval(self, dataloader: data.DataLoader) -> flcore.MetricResult:
        self.model.eval()
        self.metrics.reset()
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = loss_fn(y_hat, y)
                self.metrics.update(preds=y_hat, target=y, value=loss)
        metric_result = flcore.MetricResult(
            {name: value.item() for name, value in self.metrics.compute().items()}
        )

        return metric_result

    def evaluate(self) -> MetricResult:
        return self._eval(self.eval_dataloader)

    def test(self) -> MetricResult:
        return self._eval(self.test_dataloader)


class FedAvgServer(flcore.Server):
    pass


class FedAvg(flcore.FederatedLearning):
    def algorithm(self):
        for global_epoch in self.progress(range(self.server.max_epoch), "Global Epoch"):
            selected_clients = self.server.select_clients()

            # train local models
            local_models = []
            weights = []
            for client in self.progress(selected_clients, "Trained Client"):
                with client:
                    client.receive_model(self.server.model)
                    client.train()
                    model = client.send_model()
                    local_models.append(model)
                    weights.append(
                        client.dataset_size
                        / sum(c.dataset_size for c in selected_clients)
                    )

            self.server.aggregate(local_models, weights)

            # evaluate
            evaluation = self.server.evaluate()

            # log
            self.log(flcore.LogItem(epoch=global_epoch, metrics=evaluation))

        evaluation = self.server.test()

        # save the results
        self.log(
            flcore.LogItem(
                epoch=self.server.max_epoch,
                metrics=evaluation,
                others={
                    "global_model": self.server.model,
                },
            ),
            big_item=True,
            filename="model.pth",
        )


class CnnModel(nn.Module):
    def __init__(self, in_channels=1, num_class=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def main():
    transforms = vision.transforms.Compose(
        [
            vision.transforms.ToTensor(),
            vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = data.ConcatDataset(
        [
            vision.datasets.CIFAR10(
                "tests/data/", download=True, train=True, transform=transforms
            ),
            vision.datasets.CIFAR10(
                "tests/data/", download=True, train=False, transform=transforms
            ),
        ]
    )
    subsets = data_utils.generate_dirichlet_subsets(
        dataset=dataset, alphas=[1] * 100, min_data=40
    )
    train_test_subsets = [
        data.random_split(subset, [0.6, 0.2, 0.2]) for subset in subsets
    ]

    server = FedAvgServer(
        model=CnnModel(3, 10, 1600),
        select_ratio=0.1,
        max_epoch=100,
        learning_rate=1.0,
        robust_fn=None,
    )

    system = FedAvg(server=server, log_dir="output")

    for i in range(100):
        device = torch.device("cuda")
        client = FedAvgClient(
            id_=str(i),
            learning_rate=0.05,
            max_epoch=1,
            train_dataloader=data.DataLoader(
                train_test_subsets[i][0], batch_size=8, shuffle=True, drop_last=True
            ),
            eval_dataloader=data.DataLoader(
                train_test_subsets[i][1], batch_size=8, shuffle=True, drop_last=True
            ),
            test_dataloader=data.DataLoader(
                train_test_subsets[i][2], batch_size=8, shuffle=True, drop_last=True
            ),
            device=device,
            num_class=10,
        )
        server.register_client(client)

    system.run()


if __name__ == "__main__":
    main()
