@testset "performance.jl" begin
    @testset "MSE == prior variance" begin
        imodel = PoissonExpModel(20, 0.025, 1)
        smodel = OUModel(1, 1)
        omodel = ExpGainModel(0, 0)
        filter = FullSF(1, smodel, omodel)

        sim = Simulation(imodel, smodel, omodel, filter)
        mse = ComputeMSE(sim; num_timesteps = 100000, timestep = 0.01, burnin = 1000)

        @test 0.9 < mse < 1.1

        filter = DiagSF(1, smodel, omodel)

        sim = Simulation(imodel, smodel, omodel, filter)
        mse = ComputeMSE(sim; num_timesteps = 100000, timestep = 0.01, burnin = 1000)

        @test 0.9 < mse < 1.1

        filter = BlockSF(1, 1, smodel, omodel)

        sim = Simulation(imodel, smodel, omodel, filter)
        mse = ComputeMSE(sim; num_timesteps = 100000, timestep = 0.01, burnin = 1000)

        @test 0.9 < mse < 1.1
    end
    
    @testset "Benchmark ComputeMSE of FullSF with dim = 1024" begin
        dim = 1024

        imodel = PoissonExpModel(20, 0.025, dim)
        smodel = OUModel(100, 1)
        omodel = ExpGainModel(20., 0.01)
        filter = FullSF(dim, smodel, omodel)

        sim = Simulation(imodel, smodel, omodel, filter)

        println("")
        println("Benchmarking ComputeMSE of FullSF with dim = 1024")
        display(@benchmark ComputeMSE($sim; num_timesteps = 1000, timestep = 0.001))
        println("")
        println("")
    end

    @testset "Benchmark ComputeMSE of BlockSF with 128 8x8 blocks" begin
        dim = 1024
        numblocks = 128
        blocksize = 8

        imodel = PoissonExpModel(20, 0.025, dim)
        smodel = OUModel(100, 1)
        omodel = ExpGainModel(20., 0.01)
        filter = BlockSF(numblocks, blocksize, smodel, omodel)

        sim = Simulation(imodel, smodel, omodel, filter)

        println("")
        println("Benchmarking ComputeMSE of BlockSF with 128 8x8 blocks")
        display(@benchmark ComputeMSE($sim; num_timesteps = 1000, timestep = 0.001))
        println("")
        println("")
    end

    @testset "Benchmark ComputeMSE of DiagSF with dim = 1024" begin
        dim = 1024

        imodel = PoissonExpModel(20, 0.025, dim)
        smodel = OUModel(100, 1)
        omodel = ExpGainModel(20., 0.01)
        filter = DiagSF(dim, smodel, omodel)

        sim = Simulation(imodel, smodel, omodel, filter)

        println("")
        println("Benchmarking ComputeMSE of DiagSF with dim = 1024")
        display(@benchmark ComputeMSE($sim; num_timesteps = 1000, timestep = 0.001))
        println("")
        println("")
    end
end#performance.jl
