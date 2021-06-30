#########
# Setup #
#########

using CSV
using DataFrames
using SynapticFilter

i = parse(Int, ENV["LSB_JOBINDEX"]) # set this to a fixed number when not on the cluster

τm        = 0.025 # membrane time constant
ε0        = 1     # EPSP amplitude           <------ !!!
τ         = 400   # OU time constant
σs        = 1     # OU variance
g0        = 20    # baseline output firing rate
β         = 0.1   # fixed beta
blocksize = 10    # size of input block
numblocks = 10    # number of input blocks
τblock    = 1.0   # duration of block activation

epoch      = ceil(Int, τ)                   # duration of one epoch
num_epochs = parse.(Int, ARGS)[1]           # number of epochs to simulate 
num_burnin = parse.(Int, ARGS)[2]           # number of epochs to burn in
ts_per_sec = 100000                         # number of timesteps per seconds 
                                            # (reciprocal of timestep)
num_timesteps = num_epochs*epoch*ts_per_sec # total number of timesteps
burnin        = num_burnin*epoch*ts_per_sec # number of timesteps for burn in
timestep      = 1/ts_per_sec                # time step
dim           = blocksize * numblocks       # total input dimension

results = DataFrame(
    filter       = String[], 
    rho          = Float64[], 
    mse          = Float64[]
)
l = ReentrantLock()

##################
# Saving results #
##################

function saveresults(df)
    #"/cluster/home/ssurace/perf_sims/csv/nonsparse_scaling_vs_sparse_nonscaling"
    CSV.write("csv/input_firing_rate"  * lpad(i, 3, "0") * ".csv", df)
end#saveresults


###########################
# Performance simulations #
###########################

smodel = OUModel(τ, σs)
omodel = ExpGainModel(g0, β)
ρ_rng = [2. ^ i for i in 0:8]

@time begin # measure total time for running performance sims

Threads.@threads for ρ in ρ_rng
    # performance sims for DiagSF, loop over number of blocks in parallel
    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    filter = DiagSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for ρ = ", ρ, ": ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "DiagSF", 
                :rho          => ρ,
                :mse          => mse
            )
        )
    end
    saveresults(results)

    # performance sims for BlockSF, loop over number of blocks in parallel
    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    filter = BlockSF(numblocks, blocksize, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for ρ = ", ρ, ": ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "BlockSF", 
                :rho          => ρ,
                :mse          => mse
            )
        )
    end
    saveresults(results)

    # performance sims for FullSF, loop over number of blocks in parallel
    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    filter = FullSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for ρ = ", ρ, ": ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "FullSF", 
                :rho          => ρ,
                :mse          => mse
            )
        )
    end
    saveresults(results)
end

println("
TOTAL SIMULATION TIME: ")
end# @time

