# Numerical experiments

This folder contains all numerical experiments that were conducted for the synaptic filter paper in Julia.

For each experiment, there is a `.jl` file in this folder that specifices the experiment being performed. 
There is a corresponding `.sh` file that can be used to launch the simulation on the euler.ethz.ch cluster.
There is also an `.ipynb` file that is used to analyse the results, which would be written to CSV files within the `/csv` subfolder.
Figures can be found in `/fig`.

## Setup

In order to set up the environment, start Julia in this subfolder and type

```julia
]instantiate
```
