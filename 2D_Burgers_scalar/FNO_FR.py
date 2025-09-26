#!/usr/bin/env python
# coding: utf-8

#Importing all the necessary libraries
import os
import sys
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

import flax
from flax import linen as nn

import optax
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, List
import scipy
from tqdm import tqdm

from models_fno import FNO3d
from utils_jax import *
from utils import *

#Read the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))

#For u-velocity field
outputs_u = (jnp.array(dataset["output_samples"]))[:1000]
print(outputs_u.shape)


#Selecting training trajectories
num_trajectories = 30
selected_trajectories = np.load(f"Selected_indices_GNS_2D_Euler{num_trajectories}.npy")
print(selected_trajectories)

#Selecting a subset of the data
outputs_u_subselected = outputs_u[selected_trajectories, :, :, :]
print(f"After subselection, shape: {outputs_u_subselected.shape}")

#Free up memory
del outputs_u, dataset

Ns, Nt, Nx, Ny = outputs_u_subselected.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

inputs = outputs_u_subselected[:, 0, :, :, None]     #(Ns, Nx, Ny, Nv=1)
outputs = outputs_u_subselected[:, :, :, :, None]    #(Ns, Nt, Nx, Ny, Nv=1)

# Create coordinate grids
xspan = np.linspace(0, 1, Nx)  # spatial domain - x
yspan = np.linspace(0, 1, Ny)  # spatial domain - y
tspan = np.linspace(0, 1, Nt)  # temporal domain

# Meshgrid to create 2D coordinate arrays
[T, X, Y] = jnp.meshgrid(tspan, xspan, yspan, indexing='ij')

#T,X,Y all have (Nt, Nx, Ny)
T_tiled = jnp.tile(T[None,:,:,:], (Ns,1,1,1))
X_tiled = jnp.tile(X[None,:,:,:], (Ns,1,1,1))
Y_tiled = jnp.tile(Y[None,:,:,:], (Ns,1,1,1))

# Printing shapes after tiling across all samples
print(f"T_tiled shape: {T_tiled.shape}")
print(f"X_tiled shape: {X_tiled.shape}")
print(f"X_tiled shape: {X_tiled.shape}")

# tile inputs
inputs_tiled = jnp.tile(inputs[:,None,:,:,:], (1, Nt, 1, 1, 1))
print(f"Shape of tiled inputs: {inputs_tiled.shape}")

#Stack all
inputs_to_FNO = jnp.concatenate([inputs_tiled, T_tiled[...,None], X_tiled[...,None], Y_tiled[...,None]], axis=-1)
output_FNO = outputs
print("After dataset preparation for FNO full rollout, shapes...")
print(f"Inputs_to_FNO: {inputs_to_FNO.shape}, Outputs_from_FNO: {output_FNO.shape}")

#Free up some memory
del inputs_tiled, T_tiled, X_tiled, Y_tiled

#Separate into train and test datasets
Ntrain = int(0.8*Ns)
perm = jax.random.permutation(jax.random.PRNGKey(0), Ns)

train_idx = perm[:Ntrain]
test_idx = perm[Ntrain:]

train_x = jnp.take(inputs_to_FNO, train_idx, axis=0)
test_x = jnp.take(inputs_to_FNO, test_idx, axis=0)

train_y = jnp.take(output_FNO, train_idx, axis=0)
test_y = jnp.take(output_FNO, test_idx, axis=0)

print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")


#Stop gradients for test_x and test_y
test_x = jax.lax.stop_gradient(test_x)
test_y = jax.lax.stop_gradient(test_y)

#Free up some memory
del inputs_to_FNO, output_FNO
modes1 = 16
modes2 = 16
modes3 = 16

#Create the FNO-2D model object
fno = FNO3d(in_channels = train_x.shape[-1],
            out_channels = train_y.shape[-1],
            modes1 = modes1,
            modes2 = modes2,
            modes3 = modes3,
            width = 32,
            n_blocks = 4,
            activation = nn.activation.gelu,  
)


model_fn = jax.jit(fno.apply)

#Instantiate the model params
params = fno.init(jax.random.PRNGKey(42), train_x[0:1])


@jax.jit
def loss_fn(params, x, y):
    y_pred = model_fn(params, x)
    loss = jnp.mean((y_pred - y) ** 2)
    return loss

@jax.jit
def make_step(params, opt_state, x, y, test_x, test_y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    val_loss = loss_fn(params, test_x, test_y)
    return params, opt_state, loss, val_loss


lr = 1e-3
lr_scheduler = optax.schedules.exponential_decay(init_value=lr, transition_steps=2000, decay_rate=0.96)
optimizer = optax.adam(lr_scheduler)
opt_state = optimizer.init(params)


loss_history = []
val_loss_history = []
batch_size = 64
shuffle_key = jax.random.PRNGKey(80)
epochs = int(5e3)
min_val_loss = jnp.inf

result_dir = "./FNO_full_rollout"
filename = f"best_model_params_FNO_{num_trajectories}.pkl"

print("Starting training...")
for epoch in range(epochs):
    shuffle_key, subkey = jax.random.split(shuffle_key)
    total_loss = 0
    total_val_loss = 0
    nbatches = 0

    for batch_x, batch_y in tqdm(dataloader(subkey, train_x, train_y, batch_size),
                                desc="Training Progress"):
        batch_x = jax.lax.stop_gradient(batch_x)
        batch_y = jax.lax.stop_gradient(batch_y)

        params, opt_state, loss, val_loss = make_step(params, opt_state, batch_x, batch_y, test_x, test_y)

        total_loss += loss
        total_val_loss += val_loss
        nbatches += 1

    loss = total_loss / nbatches
    val_loss = total_val_loss / nbatches

    # if val_loss < min_val_loss:
    #     best_params = params
    #     min_val_loss = val_loss
        # save_model_params(best_params, result_dir, filename=filename)

    loss_history.append(loss)
    val_loss_history.append(val_loss)

    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Train loss: {loss}, Val loss: {val_loss}")
print("Training complete.")

plt.figure(dpi = 130)
plt.semilogy(np.arange(epoch+1), loss_history, label = "Train loss")
plt.semilogy(np.arange(epoch+1), val_loss_history, label = "Test loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tick_params(which = 'major', axis = 'both', direction = 'in', length = 6)
plt.tick_params(which = 'minor', axis = 'both', direction = 'in', length = 3.5)
plt.minorticks_on()

plt.grid(alpha = 0.3)
plt.legend(loc = 'best')

save = False
if save:
    plt.savefig(result_dir + f"/loss_plot_{num_trajectories}.jpeg", dpi = 300)
plt.show()

#Save the loss arrays
save = False
if save:
    np.save(result_dir + f"/Train_loss_{num_trajectories}.npy",loss_history)
    np.save(result_dir + f"/Test_loss_{num_trajectories}.npy",val_loss_history)

#Perform Inference
# Load the best model parameters
best_params = load_model_params(result_dir, filename = filename)

#Read the 2D Burgers' dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))

#For u-velocity field
outputs_u = (jnp.array(dataset["output_samples"]))[:1000]
print(outputs_u.shape)

Ns, Nt, Nx, Ny = outputs_u.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")


#-----Peform batched inference-------#

u0 = outputs_u[:, 0, :, :, None]     #(Ns, Nx, Ny, Nv=1)

# Shared coordinate grid (Nt, Nx, Ny, 3)
T, X, Y = jnp.meshgrid(tspan, xspan, yspan, indexing="ij")
coords = jnp.stack([T, X, Y], axis=-1)   # (Nt, Nx, Ny, 3)

def fno_forward(params, u0, coords):
    """
    u0: (B, Nx, Ny, 1)    initial condition
    coords: (Nt, Nx, Ny, 3)  shared (t,x,y) grid
    returns: (B, Nt, Nx, Ny, 1)  predicted solution
    """
    B, Nx, Ny, _ = u0.shape
    Nt = coords.shape[0]

    # Broadcast u0 along time
    u0_b = jnp.broadcast_to(u0[:, None, ...], (B, Nt, Nx, Ny, 1))  # (B, Nt, Nx, Ny, 1)

    # Add batch axis to coords
    coords_b = jnp.broadcast_to(coords[None, ...], (B, Nt, Nx, Ny, 3))  # (B, Nt, Nx, Ny, 3)

    # Concatenate along channel axis
    inputs = jnp.concatenate([u0_b, coords_b], axis=-1)  # (B, Nt, Nx, Ny, 4)

    return model_fn(params, inputs)

start_time = time.time()
print("Starting batched FR inference")
BATCH = 8
preds = []
for i in range(0, Ns, BATCH):
    u0_batch = u0[i:i+BATCH]   # (B, Nx, Ny, 1)
    pred = fno_forward(best_params, u0_batch, coords)
    preds.append(pred)
pred_FNO = jnp.concatenate(preds, axis=0)  # (Ns, Nt, Nx, Ny, 1)
print(f"FNO output prediction shape: {pred_FNO.shape}")
print("FR inference complete..")
end_time = time.time()
print(f"Inference time for {pred_FNO.shape[0]} samples: {end_time-start_time} secs")
#-----------------------------------------------#

#Compute relative L2 error
rel_l2_err_u = np.linalg.norm(pred_FNO[...,0] - outputs_u)/np.linalg.norm(outputs_u)
print(f"Overall relative L2 error with {num_trajectories}: {rel_l2_err_u}")

save = False
if save:
    np.save(result_dir + "/u_pred.npy", pred_FNO[...,0])

print("Program executed succesfully!")