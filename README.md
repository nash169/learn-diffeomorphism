# Learn Diffeomorphism
Repositories containing implementation of a NVP (Non-Volume Preserving) network for Dynamical System learning via diffeomorphic mapping.

Pytorch-based implementation of the paper: https://arxiv.org/abs/2005.13143

### Authors/Maintainers

- Bernardo Fichera (bernardo.fichera@epfl.ch)

### Run examples
In order to train a model
```sh
python(python3) -m src.examples.train_model --data <dataset_name> --model=<true|false>
(ipython) run src/examples/train_model.py --data <dataset_name>  --model=<true|false>
```
where the line commands **data** can be used to set the training dataset and **model** to load a pre-trained model. In oder to test a trained model
```sh
python(python3) -m src.examples.test_model --data <dataset_name>
(ipython) run src/examples/test_model.py --data <dataset_name>
```
where the line command **data** can be used load the model/dataset.
