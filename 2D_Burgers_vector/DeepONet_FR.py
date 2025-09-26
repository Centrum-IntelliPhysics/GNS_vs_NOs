#!/usr/bin/env python
# coding: utf-8

#importing all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
import jax
import jax.numpy as jnp
from jax import random
from jax import jit, vmap, pmap, grad, value_and_grad

import time
from tqdm import tqdm

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
from utils import *


seed = 42
np.random.seed(seed)
key = random.PRNGKey(seed)

print("Seed: ",seed)


#Load the 2D Burgers' coupled dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers_coupled"
dataset = torch.load(os.path.join(base_path, "2D_coupled_burgers.pth"))

#For u-velocity field
outputs_u = jnp.array(dataset["output_field_u"])

#For v-velocity field
outputs_v = jnp.array(dataset["output_field_v"])
print(outputs_u.shape, outputs_v.shape)

#Concatenating outputs
outputs_uv = jnp.stack([outputs_u, outputs_v], axis=-1)
print(outputs_uv.shape)


#Selecting training trajectories
num_trajectories = 30
if os.path.exists(f"subselected_train_idx_GNS_{num_trajectories}.npy"):
    print(f"Reading selected {num_trajectories} trajectories from file...")
    selected_trajectories = np.load(f"subselected_train_idx_GNS_{num_trajectories}.npy")
else:
    print(f"Selecting {num_trajectories} trajectories using PCA + KMeans")
    selected_trajectories = get_trajectory_idx(outputs_uv, num_trajectories, pca_components = 50)

#Selecting a subset of the data
outputs_uv_subselected = outputs_uv[selected_trajectories, :, :, :, :]
print(outputs_uv_subselected.shape)

#Free up memory
del outputs_u, outputs_v, outputs_uv, dataset

#Defining inputs and outputs
Ns, Nt, Nx, Ny, Nv = outputs_uv_subselected.shape

inputs = outputs_uv_subselected[:,0,:,:,:]
outputs = outputs_uv_subselected

#Define grid
tspan = jnp.linspace(0, 1, Nt)
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

#Create for trunk network
[t,x,y] = jnp.meshgrid(tspan, xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([t.flatten(), x.flatten(), y.flatten()]))
print(grid.shape)
print(grid)


# Split the data into training (2000) and testing (500) samples
inputs_train, inputs_test, outputs_train, outputs_test = \
                    train_test_split(inputs, outputs, test_size=0.2, random_state=seed)

outputs_train = outputs_train.reshape(outputs_train.shape[0], Nt*Nx*Ny, Nv)
outputs_test = outputs_test.reshape(outputs_test.shape[0], Nt*Nx*Ny, Nv)

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


#DeepONet settings
num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 2
latent_vector_size = 100

branch_network_layer_sizes = [256, 128, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*7 + [latent_vector_size]

model_u = DeepONet(branch_net_config = branch_network_layer_sizes, 
                 trunk_net_config = trunk_network_layer_sizes,
                branch_activation = nn.activation.silu,
                trunk_activation = nn.activation.silu)
model_v = DeepONet(branch_net_config = branch_network_layer_sizes, 
                 trunk_net_config = trunk_network_layer_sizes,
                branch_activation = nn.activation.silu,
                trunk_activation = nn.activation.silu)

model_fn_u = jax.jit(model_u.apply)
model_fn_v = jax.jit(model_v.apply)

@jax.jit
def loss_fn(params_u, params_v, branch_inputs, trunk_inputs, gt_outputs):
    
    #U-velocity field
    u_initial = branch_inputs[...,0]
    u_pred = model_fn_u(params_u, u_initial, trunk_inputs)
    u_true = gt_outputs[...,0]
    
    #V-velocity field
    v_initial = branch_inputs[...,1]
    v_pred = model_fn_v(params_v, v_initial, trunk_inputs)
    v_true = gt_outputs[...,1]
                             
    loss_u = jnp.mean(jnp.square(u_pred - u_true))
    loss_v = jnp.mean(jnp.square(v_pred - v_true))

    mse_loss = loss_u + loss_v
    
    return mse_loss


@jax.jit
def update(params_u, params_v, branch_inputs, trunk_inputs, gt_outputs, opt_state_u, opt_state_v):
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0,1))(params_u, params_v, 
                                              branch_inputs, trunk_inputs, gt_outputs)
    
    grads_u, grads_v = grads

    updates_u, opt_state_u = optimizer_u.update(grads_u, opt_state_u)
    updates_v, opt_state_v = optimizer_v.update(grads_v, opt_state_v)

    params_u = optax.apply_updates(params_u, updates_u)
    params_v = optax.apply_updates(params_v, updates_v)
    
    return params_u, params_v, opt_state_u, opt_state_v, loss


# Initialize model parameters
params_u = model_u.init(key, branch_inputs_train[0:1, :, :, 0], trunk_inputs_train[0:1, ...])
params_v = model_v.init(key, branch_inputs_train[0:1, :, :, 1], trunk_inputs_train[0:1, ...])

# # Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=2000, decay_rate=0.96)

optimizer_u = optax.adam(learning_rate=lr_scheduler)
optimizer_v = optax.adam(learning_rate=lr_scheduler)

opt_state_u = optimizer_u.init(params_u)
opt_state_v = optimizer_v.init(params_v)

training_loss_history = []
test_loss_history = []
# num_epochs = int(1.5e5)
num_epochs = int(2e3)
batch_size = 32

min_test_loss = jnp.inf

filepath = 'DON_FR'
filename = f"model_params_best_{num_trajectories}.pkl"

for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]
    
    # Update the parameters and optimizer state
    params_u, params_v, opt_state_u, opt_state_v, loss = update(
        params_u=params_u,
        params_v=params_v,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state_u=opt_state_u,
        opt_state_v=opt_state_v
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss = loss_fn(params_u = params_u,
                            params_v = params_v,
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    # if test_mse_loss < min_test_loss:
    #     best_params = {"params_u": params_u, "params_v":params_v}
    #     save_model_params(best_params, path = filepath, filename = filename)
    #     min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, best_test_loss: {min_test_loss}")


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

#Import the best model params
best_params = load_model_params(path = filepath, filename = filename)
best_params_u = best_params['params_u']
best_params_v = best_params['params_v']

dataset = torch.load(os.path.join(base_path, "2D_coupled_burgers.pth"))

#For u-velocity field
outputs_u = jnp.array(dataset["output_field_u"])

#For v-velocity field
outputs_v = jnp.array(dataset["output_field_v"])

Ns, Nt, Nx, Ny = outputs_u.shape
print("At inference...")
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

#Do inference
start_time = time.time()
u_pred = model_fn_u(best_params_u, outputs_u[:,0,:,:], trunk_inputs_test)
v_pred = model_fn_v(best_params_v, outputs_v[:,0,:,:], trunk_inputs_test)
end_time = time.time()

print(f"Overall inference time for u and v velocity fields: {end_time-start_time} secs")

u_pred_reshaped = u_pred.reshape(u_pred.shape[0], Nt, Nx, Ny)
v_pred_reshaped = v_pred.reshape(v_pred.shape[0], Nt, Nx, Ny)

#Compute overall relative L2 errors
rel_L2_err_u = np.linalg.norm(u_pred_reshaped - outputs_u)/np.linalg.norm(outputs_u)
rel_L2_err_v = np.linalg.norm(v_pred_reshaped - outputs_v)/np.linalg.norm(outputs_v)

print(f"Overall relative L2 error in u: {rel_L2_err_u}")
print(f"Overall relative L2 error in v: {rel_L2_err_v}")

save = False
if save:
    np.save(filepath + "/u_pred.npy", u_pred)
    np.save(filepath + "/v_pred.npy", v_pred)