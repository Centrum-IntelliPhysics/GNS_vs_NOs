#!/usr/bin/env python
# coding: utf-8

#importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap, grad, value_and_grad

import flax
import flax.linen as nn
import optax

from tqdm import tqdm
import time

from typing import Callable, Tuple, List, Dict, Optional, Any, Sequence

from sklearn.model_selection import train_test_split

from functools import partial

import os
import sys
import pickle

from models_jax import DeepONet
from utils_jax import *

seed = 42
np.random.seed(seed)
key = random.PRNGKey(seed)

print("Seed: ",seed)

#Load the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))
inputs = dataset['input_samples']
outputs = dataset['output_samples']

inputs = jnp.array(inputs)
outputs = jnp.array(outputs)

#Consider first 2500 samples out of 5000 samples
inputs = inputs[:1000, :, :]
outputs = outputs[:1000, :, :, :]

Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

print(f"Inputs: {inputs.shape}, outputs: {outputs.shape}")     

#Select only a subset of the samples
num_trajectories = 30
selected_idx = np.load(f"Selected_indices_GNS_2D_Euler{num_trajectories}.npy")
print(selected_idx)

outputs = outputs[selected_idx, ...]
inputs = inputs[selected_idx, ...]
print("After subselection")
print(f"Inputs: {inputs.shape}, outputs: {outputs.shape}")     
del dataset

#Create for trunk network
[t,x,y] = jnp.meshgrid(tspan, xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten()]))
print(grid.shape)
print(grid)

# Split the data into training (2000) and testing (500) samples
inputs_train, inputs_test, outputs_train, outputs_test = \
                    train_test_split(inputs, outputs, test_size=0.2, random_state=seed)

outputs_train = outputs_train.reshape(outputs_train.shape[0], Nt*Nx*Ny)
outputs_test = outputs_test.reshape(outputs_test.shape[0], Nt*Nx*Ny)

# Check the shapes of the subsets
print("Shape of inputs_train:", inputs_train.shape)
print("Shape of inputs_test:", inputs_test.shape)
print("Shape of outputs_train:", outputs_train.shape)
print("Shape of outputs_test:", outputs_test.shape)

#Network Inputs
branch_inputs_train = inputs_train    
trunk_inputs_train = grid             

#Inspecting the shapes
print("Shape of train branch inputs: ",branch_inputs_train.shape)
print("Shape of train trunk inputs: ",trunk_inputs_train.shape)
print("Shape of train output: ",outputs_train.shape)

#Network Inputs
branch_inputs_test = inputs_test      
trunk_inputs_test = grid              

#Inspecting the shapes
print("Shape of test branch inputs: ",branch_inputs_test.shape)
print("Shape of test trunk inputs: ",trunk_inputs_test.shape)
print("Shape of test output: ",outputs_test.shape)

num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 3
latent_vector_size = 100

branch_network_layer_sizes = [256, 128, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*7 + [latent_vector_size]

model = DeepONet(branch_net_config = branch_network_layer_sizes, 
                trunk_net_config = trunk_network_layer_sizes,
                branch_activation=nn.activation.silu,
                trunk_activation=nn.activation.silu)
model_fn = jax.jit(model.apply)

# Define the training process from here
@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs):
    predictions = model_fn(params, branch_inputs,trunk_inputs)
    mse_loss = jnp.mean(jnp.square(predictions - gt_outputs))   
    l2_error = jnp.linalg.norm(predictions - gt_outputs)/jnp.linalg.norm(gt_outputs)
    return mse_loss, l2_error

@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state):
    (loss, l2_error), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params, branch_inputs, trunk_inputs, gt_outputs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, l2_error

# Initialize model parameters
key = random.PRNGKey(seed)
params = model.init(key, branch_inputs_train, trunk_inputs_train)

# Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value = 1e-3, transition_steps = 2000, decay_rate = 0.96)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
# num_epochs = int(2e5)
num_epochs = int(2e3)
batch_size = 32

min_test_l2_error = jnp.inf
min_test_mse_loss = jnp.inf

filepath = 'DeepONet_full_rollout_compare_GNS'
filename = f"model_params_best_{num_trajectories}.pkl"

#Freeing memory by deleting inputs and outputs
del inputs, outputs

for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]

    # Update the parameters and optimizer state
    params, opt_state, loss, l2_error = update(
        params=params,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state=opt_state
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss, test_l2_error = loss_fn(params = params, 
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    if test_l2_error < min_test_l2_error:
        min_test_l2_error = test_l2_error
        
    # if test_mse_loss < min_test_mse_loss:
    #     best_params = params
    #     save_model_params(best_params, path = filepath, filename = filename)
    #     min_test_mse_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, min_test_loss: {min_test_mse_loss}, min_test_l2_error: {min_test_l2_error}")


plt.figure(dpi = 130)
plt.semilogy(np.arange(epoch+1), training_loss_history, label = "Train loss")
plt.semilogy(np.arange(epoch+1), test_loss_history, label = "Test loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tick_params(which = 'major', axis = 'both', direction = 'in', length = 6)
plt.tick_params(which = 'minor', axis = 'both', direction = 'in', length = 3.5)
plt.minorticks_on()

plt.grid(alpha = 0.3)
plt.legend(loc = 'best')

save = False
if save:
    plt.savefig(filepath + f"/loss_curves_{num_trajectories}.jpeg", dpi=200)
plt.show()

#Save the loss arrays
save = False
if save:
    np.save(filepath + "/train_loss.npy", training_loss_history)
    np.save(filepath + "/test_loss.npy", test_loss_history)

base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))
inputs = dataset['input_samples']
outputs = dataset['output_samples']

inputs = jnp.array(inputs)
outputs = jnp.array(outputs)

inputs = inputs[:1000]
outputs = outputs[:1000]

Ns, Nt, Nx, Ny = outputs.shape

#Free memory by deleting dataset
del dataset

tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

#Create for trunk network
[t,x,y] = jnp.meshgrid(tspan, xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten()]))

#Creating grid for branch inputs new and trunk_inputs_new
branch_inputs_new = inputs
trunk_inputs_new = grid

#Predictions
print("Starting inference...")

#Import the best model saved after full training
best_params = load_model_params(path = filepath, filename = filename)
start_time = time.time()
u_pred = model_fn(best_params, branch_inputs_new, trunk_inputs_new)
end_time = time.time()
print(f"Inference complete in {end_time-start_time} secs")

u_pred = u_pred.reshape(Ns, Nt, Nx, Ny)
print(f"Shapes: u_pred: {u_pred.shape}, outputs: {outputs.shape}")

#Compute overall relative L2 error
rel_l2_err = np.linalg.norm(u_pred - outputs)/np.linalg.norm(outputs)
print(f"Overall relative L2 error: {rel_l2_err}")

save=False
if save:
    np.save(filepath + "/u_pred.npy", u_pred)