module SynapticFilter

using LinearAlgebra: dot

include("models.jl")
export 
    NeuronModel, NeuronObs, OUModel,
    update!

include("filters.jl")
export 
    FilterState,
    BlockSF, DiagSF, FullSF

end#module
