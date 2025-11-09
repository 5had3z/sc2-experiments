# sc2-experiments

Repo contains ML experiments using [sc2-serializer](https://github.com/5had3z/sc2-serializer) datasets. Most experimental setups only take a few hours to run for convergence, and were trained on GTX 1080 or RTX 3090 depending on VRAM requirements for the model. Model weights, configuration and training logs from the reported experiments are provided on [Google Drive](https://drive.google.com/drive/u/1/folders/1zlDo88efK6rg-PYbslFTBF-H_TJvJ-Nr). However, some of the configurations are from an older version of the codebase so some of the keyword arguments have changed.

## Basic Repo Usage

### Training

The [train.py](./forecasting/train.py) is the main entrypoint for training all the different task and model variations. Set the environment variable `DATAPATH` to point to where the datasets are located. This repo is compatible with `torchrun` for distributed data-parallel training.  An example training (and evaluation) launch command is included in [launch.json](./.vscode/launch.json).

### Evaluating

Standard evaluation statistics generated during training must be gathered with `konduct-metadata reduce-all --workspace=$WORSPACE_HERE` before it can be displayed in `konduct-review`. Additional evaluation statistics are generated with [minimap_eval.py](./forecasting/minimap_eval.py) and [outcome_eval](./forecasting/outcome_eval.py). To view results and include the additional pages from this repo, use [review.py](./forecasting/review.py) .

## Outcome Forecasting

### Preprocessing

Reading a replay with thousands of data points to only sample 20 is a bit inefficient, and can become a large overhead when yielding a batch of 128 and training with a small neural network. Hence we instead preprocess the data and only keep the samples we are after with `./forecasting/data.py make-numpy-subset`. We use this preprocessed data to train our model with the `dali-folder` dataset type, and can change what is loaded from this subset with `keys: [win, valid, minimaps, scalars]`. [data-subset.yml](./forecasting/cfg/data-subset.yml) is an example configuration for using `make-numpy-subset`, and what we used for our results.

## Minimap Forecasting

### Preprocessing

Minimap forecasting requires finding valid subsequences of data with a regular time interval. This can be done live, by randomly sampling and validating, however there's no guarantee that you're going to spend a long time finding a good subsequence. Another preprocessing script command `./forecasting/data.py write-valid-stride-files` finds all the valid subsequences of a replay and records them. Then we can randomly sample from this set for dataloading during training, rather than sampling and testing. You can then enable `precalculated_clips: true` in your `dali-replay-clip` dataset configuration. Its also preferred to set `yields_batch: true` for faster dataloading, technically the batch is now not IID, but with and without yielding a batch has no accuracy impact.

### Git hooks

The CI will run several checks on the new code pushed to the repository. These checks can also be run locally without waiting for the CI by following the steps below:

1. [install pre-commit](https://pre-commit.com/#install),
2. Install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit. The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`.

> Note: when you commit, you may have to explicitly run `pre-commit run --all-files` twice to make it pass, as each formatting tool will first format the code and fail the first time but should pass the second time.
