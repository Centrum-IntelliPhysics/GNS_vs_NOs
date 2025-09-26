#!/usr/bin/env python
# coding: utf-8

#importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
from sklearn.preprocessing import StandardScaler

import torch
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap, grad, value_and_grad

from tqdm import tqdm
import time

import flax
import flax.linen as nn
import optax

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

#Load the 2D Allen Cahn dataset
base_path = "/data/sgoswam4/Dibya/Datasets/2D_Allen_Cahn/AllenCahn2D_32.mat"
dataset_mat = scipy.io.loadmat(base_path)
print(dataset_mat.keys())

#Consider 1st 1000 samples
outputs = jnp.array(dataset_mat['u_train'])[:1000, ...]

#Select only a subset of the samples
num_trajectories = 30
selected_idx = np.load(f"Selected_indices_GNS_2D_Euler{num_trajectories}.npy")
print(selected_idx)

outputs = outputs[selected_idx, ...]
print("After subselection")

Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")
inputs = outputs[:, 0, :, :]

del dataset_mat

tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)
print(inputs.shape, outputs.shape)

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

#Network Inputs - train
branch_inputs_train = inputs_train    
trunk_inputs_train = grid             

#Inspecting the shapes
print("Shape of train branch inputs: ",branch_inputs_train.shape)
print("Shape of train trunk inputs: ",trunk_inputs_train.shape)
print("Shape of train output: ",outputs_train.shape)

#Network Inputs - test
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
print(model)

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
lr_scheduler = optax.schedules.exponential_decay(init_value = 1e-3, 
                                                 transition_steps = 2000, decay_rate = 0.96)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
# num_epochs = int(2e5)
num_epochs = int(2e3)
batch_size = 128

min_test_l2_error = jnp.inf
min_test_mse_loss = jnp.inf

filepath = 'DeepONet_full_rollout'
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
# plt.savefig(filepath + f"/loss_plot_{num_trajectories}.jpeg", dpi = 800)
plt.close()

#Save the loss arrays
save = False
if save:
    np.save(filepath + "/train_loss.npy", training_loss_history)
    np.save(filepath + "/test_loss.npy", test_loss_history)


#Consider 1st 1000 samples
dataset_mat = scipy.io.loadmat(base_path)
outputs = jnp.array(dataset_mat['u_train'])[:1000, ...]
Ns, Nt, Nx, Ny = outputs.shape
print(outputs.shape)
inputs = outputs[:, 0, :, :]

#Free memory by deleting dataset
del dataset_mat

tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

#Create for trunk network
[t,x,y] = jnp.meshgrid(tspan, xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten()]))

#Creating grid for branch inputs new and trunk_inputs_new
branch_inputs_new = inputs
trunk_inputs_new = grid
trunk_inputs_new.shape, trunk_inputs_new

# Predictions
import time

#Import the best model saved after full training
best_params = load_model_params(path = filepath, filename = filename)

start_time = time.time()
predictions_outputs_new = model_fn(best_params, branch_inputs_new, trunk_inputs_new)
end_time = time.time()
print(f"Total time of inference for {predictions_outputs_new.shape[0]} samples: {end_time-start_time}")

print(predictions_outputs_new.shape, trunk_inputs_new.shape)

predictions_outputs_new = predictions_outputs_new.reshape(predictions_outputs_new.shape[0], Nt, Nx, Ny)

#Randomly selecting "size" number of samples out of the test dataset
random_samples = np.random.choice(np.arange(outputs.shape[0]), size = 3, replace = 'False')

t_query = [0, 25, 50, 75, -1]
for i in random_samples:
    
    for t in t_query:
        prediction_i = predictions_outputs_new[i, t, :, :]
        target_i = outputs[i, t, :, :]
        
        error_i = np.abs(prediction_i - target_i)

        plt.figure(figsize = (12,3))

        plt.subplot(1,3,1)
        contour1 = plt.contourf(xspan, yspan, prediction_i.T, levels = 20, cmap = 'viridis')
        cbar1 = plt.colorbar()
        cbar1.ax.tick_params(labelsize = 12)
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("y", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("Predicted", fontsize = 16)

        plt.subplot(1,3,2)
        contour2 = plt.contourf(xspan, yspan, target_i.T, levels = 20, cmap = 'viridis')
        cbar2 = plt.colorbar()
        cbar2.ax.tick_params(labelsize = 12)
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("y", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("Actual", fontsize = 16)


        plt.subplot(1,3,3)
        contour3 = plt.contourf(xspan, yspan, error_i.T, levels = 20, cmap = 'Wistia')
        cbar3 = plt.colorbar()
        cbar3.ax.tick_params(labelsize = 12)
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("y", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.title("Error", fontsize = 16)
        
        plt.suptitle(f"Idx: {i}, timestep: {t}")

        plt.tight_layout()
        # plt.savefig(filepath + f"/Contour_plots_{i}_timestep_{t}_{num_trajectories}.jpeg", dpi = 800)
        plt.show()

overall_rel_L2_err = jnp.linalg.norm(predictions_outputs_new - outputs)/jnp.linalg.norm(outputs)
print(f"Overall relative L2 error: {overall_rel_L2_err}")

#Save the auto_reg_error array for comparing with NODE approach
save = False
if save:
    np.save(filepath + "/u_pred.npy", predictions_outputs_new)

print("Program executed successfully!")