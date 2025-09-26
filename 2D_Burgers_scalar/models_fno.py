#Importing all the necessary libraries
import os
import sys
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import flax
from flax import linen as nn

import optax
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, List
import scipy

#Create the spectral convolution 1D class
class SpectralConv1d(nn.Module):
    in_channels: int
    out_channels: int
    modes: int

    @nn.compact
    def __call__(self, x):
        batch_size, in_channels, spatial_points = x.shape
        
        # Define trainable real and imaginary weights
        scale = 1.0 / (self.in_channels * self.out_channels)
        
        real_weights = self.param(
            "real_weights", 
            nn.initializers.uniform(scale), 
            (self.in_channels, self.out_channels, self.modes)
        )
        imag_weights = self.param(
            "imag_weights", 
            nn.initializers.uniform(scale), 
            (self.in_channels, self.out_channels, self.modes)
        )
        
        def complex_mult1d(x_hat, w):
            #x_hat (bs, in_channels, modes) * w (in_channels, out_channels, modes)
            return jnp.einsum("biM,ioM->boM", x_hat, w)    
        
        # Perform rFFT along spatial dimension
        # shape of x_hat is (batch_size, in_channels, spatial_points//2+1)
        x_hat = jnp.fft.rfft(x, axis=-1)
        
        # shape of x_hat_under_modes is (batch_size, in_channels, self.modes)
        x_hat_under_modes = x_hat[:, :, :self.modes]
        
        weights = real_weights + 1j * imag_weights
        
        # shape of out_hat_under_modes is (out_channels, self.modes)
        out_hat_under_modes = complex_mult1d(x_hat_under_modes, weights)

        # Create full frequency spectrum with zeros and insert transformed modes
        out_hat = jnp.zeros((batch_size, self.out_channels, x_hat.shape[-1]), dtype=x_hat.dtype)
        out_hat = out_hat.at[:, :, :self.modes].set(out_hat_under_modes)

        # Inverse FFT to return to spatial domain
        out = jnp.fft.irfft(out_hat, n=spatial_points, axis=-1)
        return out
    
#Create a class for implementing one FNO block or fourier layer
class FNOBlock1d(nn.Module):
    in_channels: int
    out_channels: int
    modes: int
    activation: Callable

    @nn.compact
    def __call__(self, x):
        spectral_out = SpectralConv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes=self.modes
        )(x)
        
        #Convert x into appropriate input for flax.linen.Conv
        #Shape: (batch_dim, spatial_points, channels)
        x_perm = jnp.swapaxes(x, 1, 2)
        
        bypass_out = nn.Conv(features = self.out_channels, kernel_size = 1, use_bias=True)(x_perm)
        bypass_out = jnp.swapaxes(bypass_out, 1, 2)
        
        result = self.activation(spectral_out + bypass_out)

        return result
    
#Create the main FNO-1D class
class FNO1d(nn.Module):
    in_channels: int
    out_channels: int
    modes: int
    width: int
    n_blocks: int
    activation: Callable

    @nn.compact
    def __call__(self, x):
        x_perm = jnp.transpose(x, axes=(0, 2, 1))    #(bs, spatial_points, nchannels)
        
        # Lift input to higher dimension
        x = nn.Conv(
            features=self.width,
            kernel_size=(1,),
            use_bias=True,
            name="lifting"
        )(x_perm)  # (bs, spatial_points, in_channels)
        
        x = jnp.transpose(x, axes=(0, 2, 1))

        # Apply FNO blocks
        for i in range(self.n_blocks):
            x = FNOBlock1d(
                in_channels=self.width,
                out_channels=self.width,
                modes=self.modes,
                activation=self.activation
            )(x)
        
        x_perm = jnp.transpose(x, axes=(0, 2, 1))
                               
        # Project back to desired output channels
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(1,),
            use_bias=True,
            name="projection"
        )(x_perm)

        x = jnp.transpose(x, axes=(0, 2, 1))

        return x
    

class SpectralConv2d(nn.Module):
    in_channels: int
    out_channels: int 
    modes1: int
    modes2: int

    def setup(self):
        scale = 1.0 / (self.in_channels * self.out_channels)
        
        self.weights1 = self.param(
            'weights1', lambda rng: scale * jax.random.normal(rng, 
                            (self.in_channels, self.out_channels, self.modes1, self.modes2, 2)))
        
        self.weights2 = self.param(
            'weights2', lambda rng: scale * jax.random.normal(rng, 
                            (self.in_channels, self.out_channels, self.modes1, self.modes2, 2)))

        
    def compl_mul2d(self, a, b):
        return jnp.stack([
            jnp.einsum('bixy,ioxy->boxy', a[...,0], b[...,0]) - 
            jnp.einsum('bixy,ioxy->boxy', a[...,1], b[...,1]),
            jnp.einsum('bixy,ioxy->boxy', a[...,1], b[...,0]) + 
            jnp.einsum('bixy,ioxy->boxy', a[...,0], b[...,1])
        ], axis=-1)

    def __call__(self, x):
        batchsize = x.shape[0]
        x_ft = jnp.fft.rfft2(x, norm='ortho')
        x_ft = jnp.stack([x_ft.real, x_ft.imag], axis=-1)
        
        
        out_ft = jnp.zeros((batchsize, self.out_channels, x.shape[-2], x.shape[-1]//2 + 1, 2), 
                          dtype=x.dtype)
        
        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2].set(
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1))
        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2].set(
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2))
        
        x = jnp.fft.irfft2(out_ft[...,0] + 1j * out_ft[...,1], 
                          s=(x.shape[-2], x.shape[-1]), norm='ortho')
        return x


class FNO2d(nn.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    width: int
    n_blocks: int
    activation: Callable
    
    @nn.compact
    def __call__(self, x):
        
        #x: (B, H, W, C)
        
        # Lifting: project input to higher dimension
        x = nn.Conv(features=self.width, kernel_size=(1, 1))(x)
        
        # Fourier layers with residual connections
        for i in range(self.n_blocks):

            x_permuted = jnp.transpose(x, (0, 3, 1, 2)) # (B, C, H, W)
            
            # Spectral convolution branch
            x1 = SpectralConv2d(in_channels=self.width,
                                out_channels=self.width, 
                                modes1=self.modes1, 
                                modes2=self.modes2)(x_permuted)
            x1 = jnp.transpose(x1, (0, 2, 3, 1)) # (B, H, W, C)
            # Skip connection branch (local convolution)
            x2 = nn.Conv(features=self.width, kernel_size=(1, 1))(x)
            
            # Combine branches
            x = x1 + x2
            x = self.activation(x)
        
        # Projection: map back to output space
        x = nn.Conv(features=self.out_channels, kernel_size=(1, 1))(x)
        
        return x
    

class SpectralConv3d(nn.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    modes3: int

    def setup(self):
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights1 = self.param(
            'weights1',
            lambda rng: scale * jax.random.normal(
                rng,
                (self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, 2)
            )
        )
        self.weights2 = self.param(
            'weights2',
            lambda rng: scale * jax.random.normal(
                rng,
                (self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, 2)
            )
        )
        self.weights3 = self.param(
            'weights3',
            lambda rng: scale * jax.random.normal(
                rng,
                (self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, 2)
            )
        )
        self.weights4 = self.param(
            'weights4',
            lambda rng: scale * jax.random.normal(
                rng,
                (self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, 2)
            )
        )

    def compl_mul3d(self, a, b):
        # Complex multiplication: (a_real + i a_imag) * (b_real + i b_imag)
        return jnp.stack([
            jnp.einsum('bixyz,ioxyz->boxyz', a[...,0], b[...,0]) -
            jnp.einsum('bixyz,ioxyz->boxyz', a[...,1], b[...,1]),
            jnp.einsum('bixyz,ioxyz->boxyz', a[...,1], b[...,0]) +
            jnp.einsum('bixyz,ioxyz->boxyz', a[...,0], b[...,1])
        ], axis=-1)

    def __call__(self, x):
        batchsize = x.shape[0]

        # Forward FFT
        x_ft = jnp.fft.rfftn(x, axes=(-3, -2, -1), norm="ortho")
        x_ft = jnp.stack([x_ft.real, x_ft.imag], axis=-1)   # (..., 2)

        # Output tensor in Fourier space
        out_ft = jnp.zeros(
            (batchsize, self.out_channels, x.shape[-3], x.shape[-2], x.shape[-1] // 2 + 1, 2),
            dtype=x.dtype
        )

        # Apply weights to Fourier modes
        
        #(positive k1 modes, positive k2 modes) - 1st quadrant in k1 (X-axis) and k2 (Y-axis) plane
        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2, :self.modes3].set(
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        )

        #(negative k1 modes, positive k2 modes) - 2nd quadrant in k1 (X-axis) and k2 (Y-axis) plane
        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2, :self.modes3].set(
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        )

        #(positive k1 modes, negative k2 modes) - 4th quadrant in k1 (X-axis) and k2 (Y-axis) plane
        out_ft = out_ft.at[:, :, :self.modes1, -self.modes2:, :self.modes3].set(
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        )

        #(negative k1 modes, negative k2 modes) - 3rd quadrant in k1 (X-axis) and k2 (Y-axis) plane
        out_ft = out_ft.at[:, :, -self.modes1:, -self.modes2:, :self.modes3].set(
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        )

        # Back to physical space
        x = jnp.fft.irfftn(out_ft[...,0] + 1j*out_ft[...,1],
                           s=(x.shape[-3], x.shape[-2], x.shape[-1]),
                           norm="ortho")
        return x    

class FNO3d(nn.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    modes3: int
    width: int
    n_blocks: int
    activation: Callable

    @nn.compact
    def __call__(self, x):
        # x: (B, D, H, W, C)

        # Lifting: project input to higher dimension
        x = nn.Conv(features=self.width, kernel_size=(1, 1, 1))(x)

        # Fourier layers with residual connections
        for i in range(self.n_blocks):
            # Permute to (B, C, D, H, W) for spectral conv
            x_permuted = jnp.transpose(x, (0, 4, 1, 2, 3))

            # Spectral convolution branch
            x1 = SpectralConv3d(
                in_channels=self.width,
                out_channels=self.width,
                modes1=self.modes1,
                modes2=self.modes2,
                modes3=self.modes3
            )(x_permuted)
            x1 = jnp.transpose(x1, (0, 2, 3, 4, 1))  # (B, D, H, W, C)

            # Skip connection branch (local convolution)
            x2 = nn.Conv(features=self.width, kernel_size=(1, 1, 1))(x)

            # Combine branches
            x = x1 + x2
            x = self.activation(x)

        # Projection: map back to output space
        x = nn.Conv(features=self.out_channels, kernel_size=(1, 1, 1))(x)

        return x