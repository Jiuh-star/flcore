from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchmetrics as metrics
import torchvision as vision

import flcore
import flcore.utils.data as data_utils
import flcore.utils.io as io


class FedAvgClient(flcore.ClientProtocol):
    def __init__(self, *, id_: str, model: nn.Module, learning_rate: float, max_epoch: int,
                 train_dataloader: data.DataLoader, eval_dataloader: data.DataLoader, test_dataloader: data.DataLoader,
                 device: torch.device, optimizer: optim.Optimizer, loss_fn: nn.Module, num_class: int):
        self.setup(id_=id_, model=model, learning_rate=learning_rate, max_epoch=max_epoch,
                   train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, test_dataloader=test_dataloader,
                   device=device, optimizer=optimizer, loss_fn=loss_fn)
        self.metrics = metrics.MetricCollection({
            "loss": metrics.MeanMetric(nan_strategy="error"),
            "accuracy": metrics.Accuracy("multiclass", num_classes=num_class, average="micro"),
        })
        self.metrics.to(self.device)
        self.model.to(self.device)

    def train(self, dataloader: data.DataLoader):
        self.model.train()

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            y_hat = self.model(x)

            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, dataloader: data.DataLoader) -> flcore.MetricResult:
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                self.metrics.update(preds=y_hat, target=y, value=loss)
        metric_result = flcore.MetricResult({name: value.item() for name, value in self.metrics.compute().items()})

        return metric_result


class FedAvgServer(flcore.Server):
    pass


class FedAvg(flcore.FederatedLearning):
    def __init__(self, *, server: FedAvgServer, log_dir: str | Path, checkpoints: Sequence[int] = None):
        super().__init__(server=server, log_dir=log_dir)
        self.checkpoints = checkpoints or []

    def algorithm(self):
        for global_epoch in self.progress(range(self.server.max_epoch), "Global Epoch"):
            # select client to train
            selected_clients = self.server.select_clients()

            # train local models
            for client in self.progress(selected_clients, "Trained Client"):
                client.receive_model(self.server.model)

                for local_epoch in self.progress(range(client.max_epoch), "Local Epoch"):
                    client.train(client.train_dataloader)

            self.server.aggregate(clients=selected_clients, weights=None, robust_fn=self.server.robust_fn)

            # evaluate
            evaluation = self.server.evaluate()

            # log
            self.log(flcore.LogItem(epoch=global_epoch, metrics=evaluation))

            # checkpoint
            if global_epoch in self.checkpoints:
                io.dump(self, self.log_dir / f"state-{global_epoch}.pth")

        evaluation = self.server.test()

        # save the results
        self.log(flcore.LogItem(epoch=self.server.max_epoch, metrics=evaluation, others={
            "global_model": self.server.model,
        }), big_item=True, filename="model.pth")


class CnnModel(nn.Module):
    def __init__(self, in_channels=1, num_class=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def main():
    # model = vision.models.resnet18(num_classes=10).to("cuda")
    model = CnnModel(3, 10, 1600)
    transforms = vision.transforms.Compose([
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = data.ConcatDataset([
        vision.datasets.CIFAR10("tests/data/", download=True, train=True, transform=transforms),
        vision.datasets.CIFAR10("tests/data/", download=True, train=False, transform=transforms),
    ])
    subsets = data_utils.generate_dirichlet_subsets(dataset=dataset, alphas=[1] * 20, min_data=40)
    train_test_subsets = [data.random_split(subset, [0.6, 0.2, 0.2]) for subset in subsets]

    server = FedAvgServer(select_ratio=0.1, max_epoch=800, learning_rate=1, robust_fn=None)

    system = FedAvg(server=server, log_dir="output")

    for i in range(20):
        device = torch.device("cuda")
        client_model = copy.deepcopy(model).to(device)
        optimizer = optim.SGD(client_model.parameters(), lr=0.05)
        loss_fn = nn.CrossEntropyLoss()

        client = FedAvgClient(
            id_=str(i),
            model=client_model,
            learning_rate=0.05,
            max_epoch=1,
            train_dataloader=data.DataLoader(train_test_subsets[i][0], batch_size=8, shuffle=True, drop_last=True),
            eval_dataloader=data.DataLoader(train_test_subsets[i][1], batch_size=8, shuffle=True, drop_last=True),
            test_dataloader=data.DataLoader(train_test_subsets[i][2], batch_size=8, shuffle=True, drop_last=True),
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_class=10,
        )
        server.register_client(client)

    system.run()


if __name__ == '__main__':
    main()
