module load gcc
module unload gcc

bsub -n 8 -W 24:00 -J "rho[1-100]" -R "rusage[mem=2048]" ~/julia-1.6.1/bin/julia --project=@. --threads 8 ~/exp/input_firing_rate.jl 32 8

bjobs
