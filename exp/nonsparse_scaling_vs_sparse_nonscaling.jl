#########
# Setup #
#########

using CSV
using DataFrames
using SynapticFilter

i = parse(Int, ENV["LSB_JOBINDEX"]) # set this to a fixed number when not on the cluster

ρ         = 40    # input firing rate
τm        = 0.025 # membrane time constant
ε0        = 1     # EPSP amplitude           <------ !!!
τ         = 200   # OU time constant
σs        = 1     # OU variance
g0        = 20    # baseline output firing rate
β0        = 0.1   # gain for numblocks = 1 when scaling beta with dimension
β         = 0.1   # fixed beta
blocksize = 8     # size of input block
τblock    = 1.0   # duration of block activation

# values for the number of blocks to iterate over
numblock_rng = [1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256]

epoch      = ceil(Int, τ)                   # duration of one epoch
num_epochs = parse.(Int, ARGS)[1]           # number of epochs to simulate 
num_burnin = parse.(Int, ARGS)[2]           # number of epochs to burn in
ts_per_sec = 1000                           # number of timesteps per seconds 
                                            # (reciprocal of timestep)
num_timesteps = num_epochs*epoch*ts_per_sec # total number of timesteps
burnin        = num_burnin*epoch*ts_per_sec # number of timesteps for burn in
timestep      = 1/ts_per_sec                # time step

results = DataFrame(
    filter       = String[], 
    sparse_input = Bool[], 
    beta         = Float64[], 
    dim          = Int[], 
    mse          = Float64[]
)
l = ReentrantLock()

##################
# Saving results #
##################

function saveresults(df)
    #"/cluster/home/ssurace/perf_sims/csv/nonsparse_scaling_vs_sparse_nonscaling"
    CSV.write("csv/nonsparse_scaling_vs_sparse_nonscaling"  * lpad(i, 3, "0") * ".csv", df)
end#saveresults


###########################
# Performance simulations #
###########################

smodel = OUModel(τ, σs)

@time begin # measure total time for running performance sims

Threads.@threads for numblocks in numblock_rng
    # performance sims for DiagSF
    dim = numblocks * blocksize
    β1 = β0 / sqrt(numblocks)

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β1)
    filter = DiagSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", nonsparse input:   ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "DiagSF", 
                :sparse_input => false,
                :beta         => β1,
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)

    imodel = BlockPoissonExpModel(ρ, τm, ε0, numblocks, blocksize, τblock)
    omodel = ExpGainModel(g0, β)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", sparse input: ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "DiagSF", 
                :sparse_input => true, 
                :beta         => β,
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)

    # performance sims for BlockSF
    dim = numblocks * blocksize
    β1 = β0 / sqrt(numblocks)

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β1)
    filter = BlockSF(numblocks, blocksize, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", nonsparse input:   ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "BlockSF", 
                :sparse_input => false,
                :beta         => β1, 
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)

    imodel = BlockPoissonExpModel(ρ, τm, ε0, numblocks, blocksize, τblock)
    omodel = ExpGainModel(g0, β)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", sparse input: ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "BlockSF", 
                :sparse_input => true, 
                :beta         => β,
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)

    # performance sims for FullSF
    dim = numblocks * blocksize
    β1 = β0 / sqrt(numblocks)

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β1)
    filter = FullSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", nonsparse input:   ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "FullSF", 
                :sparse_input => false,
                :beta         => β1, 
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)

    imodel = BlockPoissonExpModel(ρ, τm, ε0, numblocks, blocksize, τblock)
    omodel = ExpGainModel(g0, β)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", sparse input: ", mse)
    
    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "FullSF", 
                :sparse_input => true,
                :beta         => β, 
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)
end

println("
TOTAL SIMULATION TIME: ")
end# @time

