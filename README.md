## magma :fire:

This repo contains wrapper functions to train neural network models on pytorch and pytorch geometric (for GNNs).


### Installation 

1. Clone the repo to your local computer. 

`$ git clone https://github.com/manuflores/magma.git`

2. Install necessary libraries using `conda`.

This step assumes that you have an Anaconda installation in your computer. If you don't and want to use only python and `pip`, follow the alternative in the next section. 

We start by running the following commands: 

```
# Create environment

$ conda env create -f magma.yml python=3.7

# Activate environment

$ conda activate magma
```

Later on, you can deactivate the environment by running:

`$ conda deactivate`

Run

`$ pip install -e .`

in the root directory after cloning the repo. 

2. a. Install libraries using `pip`

Run the following command for installation using `pip` only: 

`pip install -r requirements.txt`


3. Install `torch_geometric`

Pytorch geometric is a library that enables working with graph neural networks. It's [installation is not as direct](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (hence I didn't add it to the .yml file), but based on `pip`, and depending of whether you're planning running on a CPU or GPU it has different arguments. You can run the following to install the library automatically: 

```
python -m install_torch_geom.py
```

### Examples 

