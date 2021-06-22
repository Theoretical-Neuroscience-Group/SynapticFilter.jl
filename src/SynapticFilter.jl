module SynapticFilter

using Distributions: Poisson
using LinearAlgebra: dot

include("models.jl")
export
    update!,
    State,
    NeuronObs, 
    PoissonExpModel,
    ExpGainModel, AdaptiveExpGainModel, 
    OUModel

include("filters.jl")
export 
    FilterState,
    BlockSF, DiagSF, FullSF

end#module
