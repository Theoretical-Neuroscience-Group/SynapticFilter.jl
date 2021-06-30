# Numerical experiments

This folder contains all numerical experiments that were conducted for the synaptic filter paper in Julia.

For each experiment, there is a `.jl` file in this folder that specifices the experiment being performed. 
There is a corresponding `.sh` file that can be used to launch the simulation on the euler.ethz.ch cluster.
There is also an `.ipynb` file that is used to analyse the results, which would be written to CSV files within the `/csv` subfolder.
Figures can be found in `/fig`.

## Setup (local machine)

Copy this folder (i.e. `/exp`) to your local machine.
Start Julia in this folder and type

```julia
]activate .
]instantiate
```
Replace the job index line in the `.jl` file by a fixed number.

## Setup (Euler cluster)

Download Julia 1.6.1 and unpack it into the home folder, such that the julia path is `~/julia-1.6.1/bin/julia`.

In order to set up the environment on a local machine/cluster, copy this folder (i.e. `/exp`) to the home folder (`~/`).
Then start Julia in this folder and type

```julia
]activate .
]instantiate
```
Use an `.sh` file in order to launch a simulation.
