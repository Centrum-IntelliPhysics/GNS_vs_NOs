#Import the necessary standard set of libraries

#JAX essentials
import jax, jaxlib
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax
import os, sys
import pickle

def save_model_params(params, path, filename):
    
    #Create output directory for saving model params
    if not os.path.exists(path):
        os.makedirs(path)
    
    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_model_params(path, filename):
    load_path = os.path.join(path, filename)
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return params

def dataloader(key, dataset_x, dataset_y, batch_size):
    n_samples = dataset_x.shape[0]

    n_batches = int(jnp.ceil(n_samples / batch_size))

    permutation = jax.random.permutation(key, n_samples)

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, n_samples)

        batch_indices = permutation[start:end]

        yield dataset_x[batch_indices], dataset_y[batch_indices]