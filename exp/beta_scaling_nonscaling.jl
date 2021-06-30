#########
# Setup #
#########

using CSV
using DataFrames
using SynapticFilter

i = parse(Int, ENV["LSB_JOBINDEX"]) # set this to a fixed number when not on the cluster

ρ         = 40   # input firing rate
τm        = 0.1  # membrane time constant
ε0        = 1    # EPSP amplitude           <------ !!!
τ         = 100  # OU time constant
σs        = 1    # OU variance
g0        = 20   # baseline output firing rate
β0        = 0.1  # gain for numblocks = 1
blocksize = 8    # size of input block
τblock    = 0.1  # duration of block activation

epoch      = ceil(Int, τ)                   # duration of one epoch
num_epochs = parse.(Int, ARGS)[1]           # number of epochs to simulate 
num_burnin = parse.(Int, ARGS)[2]           # number of epochs to burn in
ts_per_sec = 80000                          # number of timesteps per seconds 
                                            # (reciprocal of timestep)
num_timesteps = num_epochs*epoch*ts_per_sec # total number of timesteps
burnin        = num_burnin*epoch*ts_per_sec # number of timesteps for burn in
timestep      = 1/ts_per_sec                # time step

results = DataFrame(
    filter       = String[],
    sparse_input = Bool[],
    beta_scaling = Bool[],
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
    CSV.write("csv/beta_scaling_nonscaling"  * lpad(i, 3, "0") * ".csv", df)
end


###########################
# Performance simulations #
###########################

smodel = OUModel(τ, σs)
numblock_rng = [2^i for i in 0:7]

@time begin # measure total time for running performance sims


Threads.@threads for numblocks in numblock_rng
    # performance sims for DiagSF, loop over number of blocks in parallel
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β0)
    filter = DiagSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", beta nonscaling:   ", mse)

    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "DiagSF", 
                :sparse_input => false,
                :beta_scaling => false,
                :beta         => β0,
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)


    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", beta scaling: ", mse)

    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "DiagSF", 
                :sparse_input => false,
                :beta_scaling => true,
                :beta         => β0 / sqrt(numblocks),
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)

    # performance sims for BlockSF, loop over number of blocks in parallel
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β0)
    filter = BlockSF(numblocks, blocksize, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", beta nonscaling:   ", mse)

    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "BlockSF", 
                :sparse_input => false,
                :beta_scaling => false,
                :beta         => β0,
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)


    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", beta scaling: ", mse)

    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "BlockSF", 
                :sparse_input => false,
                :beta_scaling => true,
                :beta         => β0 / sqrt(numblocks),
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)

    # performance sims for FullSF, loop over number of blocks in parallel
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β0)
    filter = FullSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", beta nonscaling:   ", mse)

    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "FullSF", 
                :sparse_input => false,
                :beta_scaling => false,
                :beta         => β0,
                :dim          => dim, 
                :mse          => mse
            )
        )
    end
    saveresults(results)


    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", beta scaling: ", mse)

    lock(l) do
        push!(
            results, 
            Dict(
                :filter       => "FullSF", 
                :sparse_input => false,
                :beta_scaling => true,
                :beta         => β0 / sqrt(numblocks),
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

