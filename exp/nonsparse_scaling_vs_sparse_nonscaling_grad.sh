module load gcc
module unload gcc

bsub -n 8 -W 04:00 -J "grad[1-1000]" -R "rusage[mem=2048]" /cluster/home/ssurace/julia-1.6.1/bin/julia --project=@. --threads 8 /cluster/home/ssurace/perf_sims/nonsparse_scaling_vs_sparse_nonscaling_grad.jl 32 8

bjobs
