#########
# Setup #
#########

using SynapticFilter

ρ         = 40   # input firing rate
τm        = 0.1  # membrane time constant
τ         = 100  # OU time constant
σs        = 1    # OU variance
g0        = 20   # baseline output firing rate
β0        = 0.01 # gain for numblocks = 1
blocksize = 10   # size of input block
τblock    = 0.1  # duration of block activation

epoch      = ceil(Int, τ)                   # duration of one epoch
num_epochs = 16                            # number of epochs to simulate 
num_burnin = 8                              # number of epochs to burn in
ts_per_sec = 1000                           # number of timesteps per seconds 
                                            # (reciprocal of timestep)
num_timesteps = num_epochs*epoch*ts_per_sec # total number of timesteps
burnin        = num_burnin*epoch*ts_per_sec # number of timesteps for burn in
timestep      = 1/ts_per_sec                # time step

resultsFull   = Dict{Int, Float64}()
resultsBlock  = Dict{Int, Float64}()
resultsDiag   = Dict{Int, Float64}()
resultsFullS  = Dict{Int, Float64}()
resultsBlockS = Dict{Int, Float64}()
resultsDiagS  = Dict{Int, Float64}()


###########################
# Performance simulations #
###########################

smodel = OUModel(τ, σs)
numblock_rng = [2^i for i in 0:7] # 1:10

@time begin # measure total time for running performance sims

# performance sims for DiagSF, loop over number of blocks in parallel
Threads.@threads for numblocks in numblock_rng
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, dim)
    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))
    filter = DiagSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", full input:   ", mse)

    resultsDiag[numblocks] = mse


    imodel = BlockPoissonExpModel(ρ, τm, numblocks, blocksize, τblock)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", sparse input: ", mse)

    resultsDiagS[numblocks] = mse
end

# performance sims for BlockSF, loop over number of blocks in parallel
Threads.@threads for numblocks in numblock_rng
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, dim)
    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))
    filter = BlockSF(numblocks, blocksize, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", full input:   ", mse)

    resultsBlock[numblocks] = mse


    imodel = BlockPoissonExpModel(ρ, τm, numblocks, blocksize, τblock)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", sparse input: ", mse)

    resultsBlockS[numblocks] = mse
end

# performance sims for FullSF, loop over number of blocks in parallel
Threads.@threads for numblocks in numblock_rng
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, dim)
    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))
    filter = FullSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", full input:   ", mse)

    resultsFull[numblocks] = mse


    imodel = BlockPoissonExpModel(ρ, τm, numblocks, blocksize, τblock)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", sparse input: ", mse)

    resultsFullS[numblocks] = mse
end

end# @time


############
# Plotting #
############

using Plots
plot(
    resultsDiag, 
    label = "diagonal SF", 
    color = "black", 
    markershape = :circle,
    legend = :bottomright
)
plot!(
    resultsBlock, 
    label = "block SF", 
    color = "gray", 
    markershape = :circle,
    markercolor = "gray"
)
plot!(
    resultsFull, 
    label = "full SF", 
    color = "red",
    markercolor = "red",
    markershape = :circle, 
    xlabel = "number of input blocks of size 10",
    ylabel = "mean-squared error", 
)

plot!(
    resultsDiagS, 
    label = "diagonal SF, sparse input", 
    color = "black", 
    linestyle = :dash,
    markershape = :cross
)
plot!(
    resultsBlockS, 
    label = "block SF, sparse input", 
    color = "gray", 
    linestyle = :dash,
    markershape = :cross,
    markercolor = "gray"
)
plot!(
    resultsFullS, 
    label = "full SF, sparse input", 
    color = "red",
    linestyle = :dash,
    markercolor = "red",
    markershape = :cross
)

savefig("exp/fig/plot_nonsparse_HD.png")
