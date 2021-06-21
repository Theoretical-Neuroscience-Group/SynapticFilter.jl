using SynapticFilter
using BenchmarkTools, Test

@testset "SynapticFilter.jl" begin
    include("filters.jl")
    include("models.jl")
end
