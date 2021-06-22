using SynapticFilter
using BenchmarkTools, Test

const RUN_BENCHMARKS = false # optional intermediate benchmarks

@testset "SynapticFilter.jl" begin
    include("models.jl")
    include("filters.jl")
end
