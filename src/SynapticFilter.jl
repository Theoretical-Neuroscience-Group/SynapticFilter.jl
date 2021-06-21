module SynapticFilter

# using ...

include("filters.jl")
export BlockSF, DiagSF, FullSF

include("models.jl")
export NeuronModel, OUModel

end#module
