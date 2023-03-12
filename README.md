# flcore

A simple, straightforward federated learning micro framework for scholar.

## Features

- [x] Extensible, straightforward and simple structure.
- [x] Integrate some useful utilities for FL and data:
  - [io.py](flcore/utils/io.py): Atomic write so that no more afraid of interruption when writing.
  - [data.py](flcore/utils/data.py): Out-of-box data split method to generate dataset for FL.
  - [model.py](flcore/utils/model.py): Easy model parameters operation, scale gradient, map parameters etc.
  - [robust.py](flcore/utils/robust.py): Out-of-box robust aggregation function.
- [x] Beautiful progress bar and terminal logging with [rich](https://github.com/Textualize/rich).
- [x] Integrate `tensorboard` to monitor your experiment.
- [x] `LowMemoryClientMixin` to save your cuda memory.
- [x] Auto save state if keyboard interruption detected.
- [x] Auto save log & metrics of all clients while system running.

### Something under work

- [ ] Fully test.
- [ ] Well document

## How-tos

There is no tutorial for now since some API may change, but you can check [example.py](example.py).

### LICENSE

This project is open sourced and under `GNU General Public License v3.0`.
