# Video Super-Resolution Research

Research repository for the development and experimental evaluation of video super-resolution models.

This repository contains the source code, configuration files, training and evaluation utilities, and experimental components used during ongoing research.

A separate public repository may be prepared later for a curated and reproducible release associated with a publication.

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:CristianComellas21/video-super-resolution-research.git
cd video-super-resolution-research
```

### 2. Create the environment

Create the Conda environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate <environment-name>
```

The environment name is defined in `environment.yml`.

## Pretrained SpyNet weights

The project uses pretrained SpyNet weights for optical flow estimation.

The required checkpoint is included in the repository at:

```text
checkpoints/flow/spynet/spynet_sintel_final-3d2a1287.pth
```

Other checkpoints and trained model weights are intentionally excluded from Git.

## Running experiments

Experiments are configured using Hydra.

A typical execution has the form:

```bash
python main.py +experiment=<experiment>
```

Additional Hydra overrides can be provided directly from the command line, for example:

```bash
python main.py +experiment=<experiment> trainer.max_epochs=<epochs>
```

Hydra stores the resolved configuration and command-line overrides for each run, allowing experiments to be reproduced later.

## Repository structure

```text
callbacks/       Training callbacks
conf/            Hydra configuration files
criterions/      Loss functions and training criteria
datasets/        Dataset definitions and data loading code
litmodules/      PyTorch Lightning modules
models/          Model definitions
modes/           Training, validation, and inference modes
modules/         Internal model components
scripts/         Auxiliary and evaluation scripts
utils/           Utility functions
checkpoints/     Required pretrained weights
main.py          Main entry point
environment.yml  Conda environment definition
```

## Files not tracked by Git

The following files are intentionally excluded from the repository:

- Experiment logs
- Hydra run outputs
- Model checkpoints
- Trained model weights
- Generated predictions
- Intermediate experimental results
- Large datasets
- Temporary and debug files

The only checkpoint currently tracked directly by Git is the pretrained SpyNet model required by the optical flow module.