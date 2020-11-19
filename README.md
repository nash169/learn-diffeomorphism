# Learn Diffeomorphism
Repository containing implementation of a NVP (Non-Volume Preserving) network for Dynamical System learning via diffeomorphic mapping.

Pytorch-based implementation of the paper: https://arxiv.org/abs/2005.13143

### Authors/Maintainers

- Bernardo Fichera (bernardo.fichera@epfl.ch)

### Run examples
In order to train a model
```sh
python(python3) -m examples.train_model --data <dataset_name> --model=<true|false>
(ipython) run examples/train_model.py --data <dataset_name>  --model=<true|false>
```
where the line commands **data** can be used to set the training dataset and **model** to load a pre-trained model. In oder to test a trained model
```sh
python(python3) -m examples.test_model --data <dataset_name>
(ipython) run examples/test_model.py --data <dataset_name>
```
where the line command **data** can be used load the model/dataset.

### Install the package
In order to install the package in `.local` run
```sh
pip(pip3) install .
```
For local installation in the current directory
```sh
pip(pip3) install -e .
```