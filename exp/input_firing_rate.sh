module load gcc
module unload gcc

bsub -n 8 -W 24:00 -J "rho[1-100]" -R "rusage[mem=2048]" /cluster/home/ssurace/julia-1.6.1/bin/julia --project=@. --threads 8 /cluster/home/ssurace/perf_sims/input_firing_rate.jl 32 8

bjobs
