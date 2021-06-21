@testset "filters.jl" begin
    @testset "FullSF" begin
        μ = [0.1, 0.3, 0.5, -0.3]
        Σ = [   1.3  0.3  0.2  0.4;
                0.3  1.0 -0.4  0.8;
                0.2 -0.4  0.6  0.1;
                0.4  0.8  0.1  1.8;  ]

        fstate = FilterState(μ, Σ)

        smodel = OUModel(1., 1.)
        nmodel = NeuronModel(0.05, 0.1)

        x = [1, 1, 0, 1]
        y = 1
        obs = NeuronObs(x, y)
        dt = 1e-3

        filter = FullSF(4, smodel, nmodel)

        update!(fstate, filter, obs, dt)

        @test fstate.μ ≈ [0.29988898029434835, 0.5096884293090658, 0.48950055098528256, 0.00028347044152254863]
        @test fstate.Σ ≈ [1.2993977960588696 0.30139768586181315 0.20160011019705654 0.4011966940883045; 0.30139768586181315 0.9999975701549038 -0.39719988429309067 0.8003965287927197; 0.20160011019705654 -0.39719988429309067 0.6007999944901472 0.10180016529558478; 0.4011966940883045 0.8003965287927197 0.10180016529558478 1.7983950411324567]
    end

    @testset "DiagSF" begin
        using LinearAlgebra: diag, Diagonal

        for dim in 10:10:1000
            μ = rand(dim)
            μ2 = copy(μ)
            Σ = rand(dim)
            Σ2 = Matrix(Diagonal(Σ))

            fstate = FilterState(μ, Σ)
            fstate2 = FilterState(μ2, Σ2)

            smodel = OUModel(1., 1.)
            nmodel = NeuronModel(0.05, 0.1)

            x = rand(dim)
            y = 1
            obs = NeuronObs(x, y)
            dt = 1e-3

            filter = DiagSF(dim, smodel, nmodel)
            filter2 = FullSF(dim, smodel, nmodel)

            update!(fstate, filter, obs, dt)
            update!(fstate2, filter2, obs, dt)

            @test fstate.μ ≈ fstate2.μ
            @test maximum(abs.(fstate.Σ .- diag(fstate2.Σ))) < 1e-6
        end
    end

    @testset "BlockSF" begin
        for blocksize in [2, 4, 8, 16, 32]
            numblocks = 5
            dim = blocksize * numblocks

            μ = rand(dim)
            μ2 = copy(μ)

            Σ = rand(blocksize, blocksize, numblocks)
            Σ2 = zeros(dim, dim)

            for i in 1:numblocks, j in 1:blocksize, k in 1:blocksize
                Σ2[(i-1)*blocksize + j, (i-1)*blocksize + k] = Σ[j, k, i]
            end

            fstate = FilterState(μ, Σ)
            fstate2 = FilterState(μ2, Σ2)

            smodel = OUModel(1., 1.)
            nmodel = NeuronModel(0.05, 0.1)

            x = rand(dim)
            y = 1
            obs = NeuronObs(x, y)
            dt = 1e-3

            filter = BlockSF(numblocks, blocksize, smodel, nmodel)
            filter2 = FullSF(dim, smodel, nmodel)

            update!(fstate, filter, obs, dt)
            update!(fstate2, filter2, obs, dt)

            @test fstate.μ ≈ fstate2.μ
            
            for i in 1:numblocks, j in 1:blocksize, k in 1:blocksize
                @test fstate2.Σ[(i-1)*blocksize+j, (i-1)*blocksize+k] ≈ fstate.Σ[j, k, i]
            end 
        end
    end

    @testset "Benchmark FullSF" begin
        using SynapticFilter: _filter_update!

        dim = 1024
        μ  = rand(dim)
        Σ  = rand(dim, dim)
        x  = rand(dim)
        y  = 1
        g0 = 0.1
        β  = 0.01
        τ  = 1.
        σs = 1.
        dt = 1e-3

        println("")
        println("Benchmarking one filter update step for FullSF")
        display(@benchmark _filter_update!($μ, $Σ, $τ, $σs, $g0, $β, $x, $y, $dt))
        println("")
        println("")
    end

    @testset "Benchmark BlockSF" begin
        using SynapticFilter: _filter_update!

        dim = 1024
        numblocks = 128
        blocksize = 8
        μ  = rand(dim)
        Σ  = rand(blocksize, blocksize, numblocks)
        x  = rand(dim)
        y  = 1
        g0 = 0.1
        β  = 0.01
        τ  = 1.
        σs = 1.
        dt = 1e-3

        println("")
        println("Benchmarking one filter update step for BlockSF")
        display(@benchmark _filter_update!($μ, $Σ, $τ, $σs, $g0, $β, $x, $y, $dt))
        println("")
        println("")
    end

    @testset "Benchmark DiagSF" begin
        using SynapticFilter: _filter_update!

        dim = 1024
        μ  = rand(dim)
        Σ  = rand(dim)
        x  = rand(dim)
        y  = 1
        g0 = 0.1
        β  = 0.01
        τ  = 1.
        σs = 1.
        dt = 1e-3

        println("")
        println("Benchmarking one filter update step for DiagSF")
        display(@benchmark _filter_update!($μ, $Σ, $τ, $σs, $g0, $β, $x, $y, $dt))
        println("")
        println("")
    end

    @testset "Benchmark FullSF" begin
        using SynapticFilter: _filter_update!
        using CUDA

        dim = 1024
        μ  = CUDA.rand(dim)
        Σ  = CUDA.rand(dim, dim)
        x  = CUDA.rand(dim)
        y  = 1
        g0 = 0.1f0
        β  = 0.01f0
        τ  = 1f0
        σs = 1f0
        dt = 1f-3

        println("")
        println("Benchmarking one filter update step for FullSF on GPU")
        display(
            @benchmark CUDA.@sync _filter_update!($μ, $Σ, $τ, $σs, $g0, $β, $x, $y, $dt)
        )
        println("")
        println("")
    end
end
