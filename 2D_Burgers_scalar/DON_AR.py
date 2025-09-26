import os, sys, pickle
import jax, jaxlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
import numpy as np

import torch
import flax
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from typing import Callable, Sequence

from tqdm import tqdm
import time

from models_jax import DeepONet
from utils_jax import *
from utils import *

seed = 42
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

#Load the 2D Burgers' scalar dataset
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

#Creating the input and output training data
init_timestep = 0
end_timestep = Nt-1

input_data_NN = outputs_u_subselected[:, init_timestep, :, :]    
output_data_NN = outputs_u_subselected[:, init_timestep+1, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs_u_subselected[:,i,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs_u_subselected[:,i+1,:,:]))
print(input_data_NN.shape, output_data_NN.shape)


#Reshaping the output_data_NN from (ns*nt//2, nx, ny) to (ns*nt//2, nx*ny)
#Input_data_NN remains as it is, i.e., (ns*nt//2, nx, ny)
output_data_NN = output_data_NN.reshape(output_data_NN.shape[0], 
                                        output_data_NN.shape[1]*output_data_NN.shape[2])
print(input_data_NN.shape, output_data_NN.shape)

input_data_NN_train, input_data_NN_test, output_data_NN_train, output_data_NN_test = \
                        train_test_split(input_data_NN, output_data_NN, test_size = 0.2, random_state = 42)
print(input_data_NN_train.shape, input_data_NN_test.shape, 
      output_data_NN_train.shape, output_data_NN_test.shape)

#Freeing memory by deleting input_data_NN and output_data_NN
del input_data_NN, output_data_NN

#Form branch and trunk inputs train
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

#Create for trunk network
[x,y] = jnp.meshgrid(xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([x.flatten(), y.flatten()]))
print(grid.shape)
print(grid)

#Creating the training data for branch and trunk inputs
branch_inputs_train = input_data_NN_train
trunk_inputs_train = grid
outputs_train = output_data_NN_train

print("Shape of branch inputs train: ",branch_inputs_train.shape)
print("Shape of trunk inputs train: ",trunk_inputs_train.shape)
print("Shape of outputs train: ",outputs_train.shape)
print("Shape of grid: ",grid.shape)

#For branch and trunk inputs test
branch_inputs_test = input_data_NN_test
trunk_inputs_test = grid
outputs_test = output_data_NN_test

print("Shape of branch inputs test: ",branch_inputs_test.shape)
print("Shape of trunk inputs test: ",trunk_inputs_test.shape)
print("Shape of outputs test: ",outputs_test.shape)
print("Shape of grid: ",grid.shape)


#DeepONet settings
num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 2
latent_vector_size = 100

branch_network_layer_sizes = [256, 128, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*7 + [latent_vector_size]

model = DeepONet(branch_net_config = branch_network_layer_sizes, 
                 trunk_net_config = trunk_network_layer_sizes,
                branch_activation = nn.activation.tanh,
                trunk_activation = nn.activation.tanh)

model_fn = jax.jit(model.apply)
print(model)

@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs):
    
    u_curr = branch_inputs  # Current state input (e.g., u(t))
    u_next = gt_outputs     # Ground truth next state (e.g., u(t+1))
    
    #Without time integrator use DeepONet to predict next timestep
    u_pred_next = model_fn(params, u_curr, trunk_inputs)

    # Compute the Mean Squared Error loss between the predicted and ground truth next states
    mse_loss = jnp.mean(jnp.square(u_pred_next - u_next))
    
    return mse_loss

@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, branch_inputs, trunk_inputs, gt_outputs)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss


# Initialize model parameters
key = jax.random.PRNGKey(seed)
params = model.init(key, branch_inputs_train[0:1, ...], trunk_inputs_train[0:1, ...])

# # Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.95)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
num_epochs = int(2e5)
# num_epochs = int(2e3)
batch_size = 32

min_test_loss = jnp.inf

filepath = 'DeepONet_Autoregressive'
filename = f'model_params_best_{num_trajectories}_new.pkl'

print("Starting training...")
# for epoch in tqdm(range(num_epochs), desc="Training Progress"):
for epoch in range(num_epochs):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]

    # Update the parameters and optimizer state
    params, opt_state, loss = update(
        params=params,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state=opt_state
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss = loss_fn(params = params, 
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    if test_mse_loss < min_test_loss:
        best_params = params
        save_model_params(best_params, path = filepath, filename = filename)
        min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, best_test_loss: {min_test_loss}")
print("Training completed..")

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

save = True

if save:
    plt.savefig(filepath + f"/loss_curves_{num_trajectories}_new.jpeg", dpi=200)
plt.close()


#Save the loss arrays
save = False

if save:
    np.save(filepath + "/Train_loss.npy",training_loss_history)
    np.save(filepath + "/Test_loss.npy",test_loss_history)

@jax.jit
def inference(u_curr, trunk_inputs_test):
    u_next = model_fn(best_params, u_curr, trunk_inputs_test)
    return u_next

def run_inference(initial_u, trunk_inputs_test, n_steps):
    u_states = np.zeros(shape = (initial_u.shape[0], Nt, Nx, Ny))  # List to store the states over time
    u_states[:,0,:,:] = initial_u    #(ns, nt, nx, ny)
    
    # Initialize the previous state (this could be your u_0 and u_1, etc.)
    u_curr = initial_u  # Set the current state to the initial state   -> (n, nx, ny)
    
    for i in range(1, n_steps):
        # Perform one inference step using the multistep method
        u_next = inference(u_curr, trunk_inputs_test)
        
        #u_next is (n, nx*ny), so reshape it to (n, nx, ny)
        u_next = u_next.reshape(u_next.shape[0], Nx, Ny)
        
        # Append the predicted state to the list
        u_states[:, i, :, :] = u_next
        
        # Update previous and current states for the next step
        u_curr = u_next
    
    return u_states

# Load the best model parameters
import time

best_params = load_model_params(path=filepath, filename=filename)

#Load the 2D Burgers' scalar dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))

#For u-velocity field
outputs = (jnp.array(dataset["output_samples"]))[:1000]

Ns, Nt, Nx, Ny = outputs.shape

del dataset
u_curr = outputs[:, 0, :, :]

start_time = time.time()
u_pred = run_inference(u_curr, trunk_inputs_test, n_steps=Nt)
end_time = time.time()

print(f"Total time of inference for {u_pred.shape[0]} samples: ",end_time-start_time)
print(u_pred.shape, outputs.shape)

L2_err = jnp.linalg.norm(u_pred - outputs)/jnp.linalg.norm(outputs)
print(f"Overall rel L2 error: {L2_err}")

#Save the auto_reg_error array for comparing with NODE approach
#Saving the u_pred and ground truth output arrays for separate postprocessing

save = False
if save:
    np.save(filepath + "/u_pred.npy", u_pred)