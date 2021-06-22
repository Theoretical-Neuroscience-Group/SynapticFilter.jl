# SynapticFilter

[![Build Status](https://travis-ci.com/Theoretical-Neuroscience-Group/SynapticFilter.jl.svg?branch=master)](https://travis-ci.com/Theoretical-Neuroscience-Group/SynapticFilter.jl)
[![Coverage](https://codecov.io/gh/Theoretical-Neuroscience-Group/SynapticFilter.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Theoretical-Neuroscience-Group/SynapticFilter.jl)

This is a high-performance Julia implementation of parts of [this Python repo](https://github.com/Theoretical-Neuroscience-Group/synaptic_filter).

## Installation

To install this package in Julia 1.6, type

```julia
]add https://github.com/Theoretical-Neuroscience-Group/SynapticFilter.jl
```

## Usage

A basic performance simulation (outputting MSE) can be performed as follows

```julia
using SynapticFilter

dim = 1024
numblocks = 128
blocksize = 8

imodel = PoissonExpModel(dim, 0.025, dim)
smodel = OUModel(100, 1)
omodel = ExpGainModel(20., 0.01)

filter = FullSF(dim, smodel, omodel)
#filter = DiagSF(dim, smodel, omodel)
#filter = BlockSF(numblocks, blocksize, smodel, omodel)

sim = Simulation(imodel, smodel, omodel, filter)

ComputeMSE(sim; num_timesteps = 100000, timestep = 0.001, burnin = 10000)
```
