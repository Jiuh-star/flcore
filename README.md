# flcore

A simple, straightforward federated learning micro framework for scholar.

## Features

- [x] Extensible, straightforward and simple structure.
- [x] Integrate some utilities: `data split`, `model parameters operation` & `robust aggragation function`.
- [x] Beautiful progress bar and terminal logging, thanks to [rich](https://github.com/Textualize/rich).
- [x] Integrate `tensorboard` to monitor your experiment.
- [x] Only selected clients are load in memory in `low memory mode`.
- [x] Quick resume state in `replay mode`.

### Something under work

- [ ] Auto command line generate.
- [ ] Fully test.
- [ ] Well document

## How-tos

There is no tutorial for now since some API may change, but you can see [tests/fedavg.py](tests/fedavg.py).


### LICENSE

This project is open sourced and under `GNU General Public License v3.0`.
