#########
# Setup #
#########

using SynapticFilter

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
num_epochs = 32                             # number of epochs to simulate 
num_burnin = 2                              # number of epochs to burn in
ts_per_sec = 10000                          # number of timesteps per seconds 
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
numblock_rng = [2^i for i in 0:4] # 1:10

@time begin # measure total time for running performance sims

# performance sims for DiagSF, loop over number of blocks in parallel
Threads.@threads for numblocks in numblock_rng
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β0)
    filter = DiagSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", beta nonscaling:   ", mse)

    resultsDiag[numblocks] = mse


    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("DiagSF for numblocks = ", numblocks, ", beta scaling: ", mse)

    resultsDiagS[numblocks] = mse
end

# performance sims for BlockSF, loop over number of blocks in parallel
Threads.@threads for numblocks in numblock_rng
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β0)
    filter = BlockSF(numblocks, blocksize, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", beta nonscaling:   ", mse)

    resultsBlock[numblocks] = mse


    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("BlockSF for numblocks = ", numblocks, ", beta scaling: ", mse)

    resultsBlockS[numblocks] = mse
end

# performance sims for FullSF, loop over number of blocks in parallel
Threads.@threads for numblocks in numblock_rng
    dim = numblocks * blocksize

    imodel = PoissonExpModel(ρ, τm, ε0, dim)
    omodel = ExpGainModel(g0, β0)
    filter = FullSF(dim, smodel, omodel)

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", beta nonscaling:   ", mse)

    resultsFull[numblocks] = mse


    omodel = ExpGainModel(g0, β0 / sqrt(numblocks))

    sim = Simulation(imodel, smodel, omodel, filter)

    mse = ComputeMSE(
        sim; num_timesteps = num_timesteps, timestep = timestep, burnin = burnin
    )
    println("FullSF for numblocks = ", numblocks, ", beta scaling: ", mse)

    resultsFullS[numblocks] = mse
end

println("
TOTAL SIMULATION TIME: ")
end# @time


############
# Plotting #
############

using Plots; pgfplotsx()
using LaTeXStrings
plot(
    resultsDiag, 
    label = "diagonal SF, β constant", 
    color = "black",    
    markershape = :circle,
    legend = :topleft,
    xscale = :log2,
    xticks = numblock_rng,
    ylims = (0, 1),
    yticks = 0:0.1:1
)
plot!(
    resultsBlock, 
    label = "block SF, β constant", 
    color = "gray", 
    markerstrokecolor = "gray",
    markershape = :circle
)
plot!(
    resultsFull, 
    label = "full SF, β constant", 
    color = "red",
    markerstrokecolor = "red",
    markershape = :circle, 
    xlabel = "number of input blocks of size 8",
    ylabel = "mean-squared error" 
)

plot!(
    resultsDiagS, 
    label = L"diagonal SF, $\beta$ scaled with $n^{-1/2}$", 
    color = "black", 
    linestyle = :dash,
    markershape = :cross
)
plot!(
    resultsBlockS, 
    label = L"block SF, $\beta$ scaled with $n^{-1/2}$", 
    color = "gray", 
    linestyle = :dash,
    markershape = :cross,
    markerstrokecolor = "gray",
)
plot!(
    resultsFullS, 
    label = L"full SF, $\beta$ scaled with $n^{-1/2}$", 
    color = "red",
    linestyle = :dash,
    markerstrokecolor = "red",
    markershape = :cross
)

savefig("exp/fig/beta_scaling_nonscaling_e0=1_hiD2.png")
