@testset "performance.jl" begin
    @testset "basic method" begin
        w = [1, 2, 5, 2]
        μ = [0, -1, 4, 2]

        @test ComputeMSE(w, μ) ≈ (1 + 3^2 + 1) / 4
    end#ComputeMSE

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
    end#ComputeMSE
end#performance.jl
