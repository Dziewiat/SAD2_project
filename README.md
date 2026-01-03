# (OPTION 1) Environment confguration  (IGNORE IF U ARE USING MacOS)
As BNfinder2 runs on Python 2 we have to set up the environment.

## Install pyenv
Install build dependencies:
```bash
sudo apt update
sudo apt install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  wget curl llvm \
  libncurses5-dev libncursesw5-dev \
  xz-utils tk-dev \
  libffi-dev
```

Download and install pyenv:
```bash
curl https://pyenv.run | bash
```

Add pyenv to `~/.bashrc`:
```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

Reload shell:
```bash
exec $SHELL
```

## Install Python 2.7
Install python2.7.18:
```bash
pyenv install 2.7.18
```

Set up your local python version in your project directory:
```bash
cd project/dir/name
pyenv local 2.7.18
python --version
```
Version check should return `2.7.18`. Also, a `.python-version` file should be created in your directory.

## Set up virtual environment with virtualenv (compatible with python2)
Install `virtualenv` package with `pip` (if not already installed):
```bash
pip install virtualenv
```

Create and activate the virtual environment with python2 of chosen name (here `.venv2`):
```bash
virtualenv -p python2 .venv2
source .venv2/bin/activate
```

## Install BNfinder
Inside of the virtual environment:
```bash
pip install BNfinder
```

Check if it worked:
```bash
bnf --help
```

Inside your python2.7 environment install requirements listed in `requirements.txt` file.


# (OPTION 2) Environment confguration on macos using Docker

## Prerequisites

Docker Desktop for macOS:
https://www.docker.com/products/docker-desktop/

Verify Docker is installed:

```bash
docker --version

# Build image

docker build --platform=linux/amd64 -t bnfinder2 .


# Run container from image

docker run -it --rm --platform=linux/amd64 \
  -v "$PWD":/work -w /work \
  bnfinder2


# Final checks
python2.7 --version

# Should print something like: Python 2.7.18

bnf --help

# Should print options for BNfinder library
```

# Generate boolean network trajectories
The script is used to generate a random boolean network and sample a random trajectory dataset from it. For usage options run:
```bash
python2.7 scripts/generate_bn_trajectory_dataset.py --help
```

Example usage with random boolean network:
```bash
python2.7 scripts/generate_bn_trajectory_dataset.py -n 5 -o generated_trajectories/test1.txt -g graphs/test1.txt
```

Example usage with a predefined network structure (`.bnet` file):
```bash
python2.7 scripts/generate_bn_trajectory_dataset.py -b models/test_model.bnet -o generated_trajectories/test1.txt -g graphs/test1.txt
```

## Advanced usage
Additionally, the boolean network with all of its methods is implemented as the `BN()` class in the script, so that it can be used in other scripts, e.g. for comparison of different boolean networks.

Constructing a `BN()` object with a higher number of nodes (>12) can be time consuming. When computing multiple trajectory datasets from one boolean network architecture, it is recomended to first: construct the `BN()` object, second: sample multiple datasets using the same `BN()` instance with `BN.generate_trajectory_dataset()` method.

# Generate multiple trajectory datasets from a parameter grid
The script is used to generate multiple trajectory datasets from a parameter grid. The simplest method for now is to modify the grid at the end of `scripts/generate_bn_trajectories_from_grid.py` and `OUTPUT_DIR` variable and run:
```bash
python2.7 scripts/generate_bn_trajectories_from_grid.py 
```

Generating networks with `n_nodes` > 12 can take a few minutes (exponential complexity of generating a state transition system), so it is recomended you make yourself some coffee after initializing the script.