module load gcc
module unload gcc

bsub -n 8 -W 04:00 -J "sparse[1-100]" -R "rusage[mem=2048]" ~/julia-1.6.1/bin/julia --project=@. --threads 8 ~/exp/nonsparse_scaling_vs_sparse_nonscaling.jl 32 8

bjobs
