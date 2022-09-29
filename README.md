# Variational Autoencoder Bayesian Matrix Factorization (VABMF)
Completed in 2020 
Tensorflow prototypes of:
* "Variational Autoencoder Bayesian Matrix Factorization" (VABMF) model (https://link.springer.com/article/10.1007/s10489-020-02049-9).

See paper ("Variational Autoencoder Bayesian Matrix Factorization for collaborative filtering") here: https://link.springer.com/article/10.1007/s10489-020-02049-9.

## Dependencies
This project was written to be compatible with Python 3.5. See `requirements.txt` for third party dependencies.

## Scripts
The `scripts/` folder contains the following Python scripts:
- `DATA_PREPROCESSING_10m,1m, and 100k`: for preprocessing up the 100k,1M,10M datasets   
- `main.py`: for training, hyperparameter selection (random search) and testing.
- main script can be invoked with the `--help` flag for more information.

## Data
The [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) was used for this project - see the `data/ml-100k/` folder. 
(The 1M and 10M Datasets were also used, which can be found [here](https://grouplens.org/datasets/movielens/1M/)(https://grouplens.org/datasets/movielens/10M/)).
