module SynapticFilter

using CUDA: AnyCuMatrix
using Distances: msd
using Distributions: Poisson
using LinearAlgebra: dot, diagind
using Statistics: mean

include("models.jl")
export
    update!,
    State,
    NeuronObs, 
    PoissonExpModel, BlockPoissonExpModel,
    ExpGainModel, AdaptiveExpGainModel, 
    OUModel

include("filters.jl")
export 
    FilterState,
    BlockSF, DiagSF, FullSF

include("simulation.jl")
export 
    Simulation, SimulationState

include("performance.jl")
export 
    ErrorMeasure, ComputeMSE

end#module
