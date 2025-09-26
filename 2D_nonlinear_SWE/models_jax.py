#Import the necessary standard set of libraries

#JAX essentials
import jax, jaxlib
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax
from typing import Callable, Sequence

class branch_net(nn.Module):

    layer_sizes: Sequence[int] 
    activation: Callable
    
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
        
        # #x has shape (ns, nx, ny) - so add channel dimension: (ns, nx, ny, nc)
        if x.ndim != 4:
            x = x[..., jnp.newaxis]
        
        #Convolutional layers
        x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")
        
        x = nn.Conv(features = 64, kernel_size = (2, 2), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape = (2,2), strides = (2,2), padding = "SAME")
        
        x = x.flatten()   #flatten
        
        #MLP layers
        for i, layer in enumerate(self.layer_sizes[:-1]):
            x = nn.Dense(layer, kernel_init = init)(x)
            x = self.activation(x)
        x = nn.Dense(self.layer_sizes[-1], kernel_init = init)(x)
        return x
    
class trunk_net(nn.Module):
    trunk_layer_config: Sequence[int]
    activation: Callable
    
    @nn.compact
    def __call__(self, x):
        
        init = nn.initializers.glorot_normal()
        
        #Branch network forward pass
        for i, layer_size in enumerate(self.trunk_layer_config):
            x = nn.Dense(layer_size, kernel_init = init)(x)
            x = self.activation(x)
        
        return x
    
class DeepONet(nn.Module):

    branch_net_config: Sequence[int]
    trunk_net_config: Sequence[int]
    branch_activation: Callable
    trunk_activation: Callable

    def setup(self):

        self.branch_net = branch_net(self.branch_net_config, self.branch_activation)
        self.trunk_net = trunk_net(self.trunk_net_config, self.trunk_activation)


    def __call__(self, x_branch, x_trunk):
        
        #Vectorize over multiple samples of input functions
        branch_outputs = jax.vmap(self.branch_net, in_axes = 0)(x_branch)
        
        #Vectorize over multiple query points
        trunk_outputs = jax.vmap(self.trunk_net, in_axes = 0)(x_trunk)       
        
        inner_product = jnp.einsum('ik,jk->ij', branch_outputs, trunk_outputs)

        return inner_product
    
class DeepONet_with_forcing(nn.Module):
    
    input_branch_net_config: Sequence[int]
    forcing_branch_net_config: Sequence[int]
    trunk_net_config: Sequence[int]
    input_branch_activation: Callable
    forcing_branch_activation: Callable
    trunk_activation: Callable

    def setup(self):

        self.input_branch_net = branch_net(self.input_branch_net_config, self.input_branch_activation)
        self.forcing_branch_net = branch_net(self.forcing_branch_net_config, self.forcing_branch_activation)
        self.trunk_net = trunk_net(self.trunk_net_config, self.trunk_activation)

    def __call__(self, x_branch_input, x_branch_forcing, x_trunk):

        #Vectorize over multiple samples of input functions
        branch_outputs_in = jax.vmap(self.input_branch_net, in_axes = 0)(x_branch_input)

        #Vectorize over multiple samples of forcing functions
        branch_outputs_forcing = jax.vmap(self.forcing_branch_net, in_axes = 0)(x_branch_forcing)

        #Add forcing and input fields (latent representations of both)
        branch_resultant = branch_outputs_in + branch_outputs_forcing

        #Vectorize over multiple query points
        trunk_outputs = jax.vmap(self.trunk_net, in_axes = 0)(x_trunk)

        inner_product = jnp.einsum('ik,jk->ij', branch_resultant, trunk_outputs)

        return inner_product