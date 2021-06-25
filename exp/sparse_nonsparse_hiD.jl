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
blocksize = 8    # size of input block
τblock    = 0.1  # duration of block activation

epoch      = ceil(Int, τ)                   # duration of one epoch
num_epochs = 128                            # number of epochs to simulate 
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

println("
TOTAL SIMULATION TIME: ")
end# @time


############
# Plotting #
############

using Plots; pgfplotsx()
plot(
    resultsDiag, 
    label = "diagonal SF", 
    color = "black",    
    markershape = :circle,
    legend = :bottomright,
    xscale = :log2,
    xticks = numblock_rng,
    ylims = (0, 1),
    yticks = 0:0.1:1
)
plot!(
    resultsBlock, 
    label = "block SF", 
    color = "gray", 
    markerstrokecolor = "gray",
    markershape = :circle
)
plot!(
    resultsFull, 
    label = "full SF", 
    color = "red",
    markerstrokecolor = "red",
    markershape = :circle, 
    xlabel = "number of input blocks of size 8",
    ylabel = "mean-squared error" 
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
    markerstrokecolor = "gray",
)
plot!(
    resultsFullS, 
    label = "full SF, sparse input", 
    color = "red",
    linestyle = :dash,
    markerstrokecolor = "red",
    markershape = :cross
)

savefig("exp/fig/sparse_nonsparse_hiD.png")

resultsDiag[1] = 0.2601673108827467
resultsDiag[2] = 0.3966687801368161
resultsDiagS[1] = 0.26782323678503567
resultsDiag[4] = 0.49670215028163955
resultsDiagS[2] = 0.40597316888846924
resultsDiagS[4] = 0.5679283192541236
resultsDiag[8] = 0.6424592953818584
resultsDiagS[8] = 0.7681252953006376
resultsDiagSF[16] = 0.7552139548551479
resultsDiagS[16] = 0.8838554924651764
resultsDiag[32] = 0.8548628511695536
resultsDiagS[32] = 0.9561353460554867
resultsDiag[64] = 0.9126570741163882
resultsDiagS[64] = 0.9836768689299181
resultsDiag[128] = 0.9550061678789515
resultsDiagS[128] = 0.9954426494747298
resultsBlock[1] = 0.2156723007351738
resultsBlock[2] = 0.3198066644728403
resultsBlockS[1] = 0.1862129671305838
resultsBlockS[2] = 0.34697504267143275
resultsBlock[4] = 0.418177739260888
resultsBlockS[4] = 0.5369980767528025
resultsBlock[8] = 0.5320928780895958
resultsBlockS[8] = 0.7395659863037587
resultsBlock[16] = 0.6351371101648544
resultsBlockS[16] = 0.8861266619256182
resultsBlock[32] = 0.7639850211049789
resultsBlockS[32] = 0.9605318488554098
resultsBlock[64] = 0.8531556090570227
resultsBlockS[64] = 0.9848938168160051
resultsBlock[128] = 0.9132110252540951
resultsBlockS[128] = 0.9935509726243488
resultsFull[1] = 0.20142420879646247
resultsFull[2] = 0.2742953790599208
resultsFullS[1] = 0.18646626871282518
resultsFullS[2] = 0.33465317296387664
resultsFull[4] = 0.37244410525023736
resultsFullS[4] = 0.519013327567376
resultsFull[8] = 0.49283225014717347
resultsFullS[8] = 0.7503495919688189
resultsFull[16] = 0.6069734723574606
resultsFullS[16] = 0.8953430874335794
resultsFull[32] = 0.704067043722794
resultsFullS[32] = 0.9579965597731406
resultsFull[64] = 0.7887769848033332
resultsFullS[64] = 0.986572562003516
resultsFull[128] = 0.8826750416811971
resultsFullS[128] = 0.999694821910929
