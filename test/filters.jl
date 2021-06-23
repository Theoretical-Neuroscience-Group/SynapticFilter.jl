@testset "filters.jl" begin
    @testset "FullSF" begin
        μ = [0.1, 0.3, 0.5, -0.3]
        Σ = [   1.3  0.3  0.2  0.4;
                0.3  1.0 -0.4  0.8;
                0.2 -0.4  0.6  0.1;
                0.4  0.8  0.1  1.8;  ]

        fstate = FilterState(μ, Σ)

        smodel = OUModel(1., 1.)
        omodel = ExpGainModel(0.05, 0.1)

        x = [1, 1, 0, 1]
        y = 1
        obs = NeuronObs(x, y)
        dt = 1e-3

        filter = FullSF(4, smodel, omodel)

        update!(fstate, filter, obs, dt)

        @test fstate.μ ≈ [0.2998895344899539, 0.5096890112144515, 0.4895005232755023, 0.00028430173493088073]
        @test fstate.Σ ≈ [1.2993979068979908 0.2993978022428903 0.19960010465510047 0.3991968603469862; 0.2993978022428903 0.9999976923550348 -0.3991998901121445 0.7983967033643355; 0.19960010465510047 -0.3991998901121445 0.600799994767245 0.0998001569826507; 0.3991968603469862 0.7983967033643355 0.0998001569826507 1.7983952905204792]
    end#FullSF

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
            omodel = ExpGainModel(0.05, 0.1)

            x = rand(dim)
            y = 1
            obs = NeuronObs(x, y)
            dt = 1e-3

            filter = DiagSF(dim, smodel, omodel)
            filter2 = FullSF(dim, smodel, omodel)

            update!(fstate, filter, obs, dt)
            update!(fstate2, filter2, obs, dt)

            @test fstate.μ ≈ fstate2.μ
            @test maximum(abs.(fstate.Σ .- diag(fstate2.Σ))) < 1e-6
        end
    end#DiagSF

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
            omodel = ExpGainModel(0.05, 0.1)

            x = rand(dim)
            y = 1
            obs = NeuronObs(x, y)
            dt = 1e-3

            filter = BlockSF(numblocks, blocksize, smodel, omodel)
            filter2 = FullSF(dim, smodel, omodel)

            update!(fstate, filter, obs, dt)
            update!(fstate2, filter2, obs, dt)

            @test fstate.μ ≈ fstate2.μ
            
            for i in 1:numblocks, j in 1:blocksize, k in 1:blocksize
                @test fstate2.Σ[(i-1)*blocksize+j, (i-1)*blocksize+k] ≈ fstate.Σ[j, k, i]
            end 
        end
    end#BlockSF

    if RUN_BENCHMARKS
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
    end#Benchmark FullSF

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
    end#Benchmark BlockSF

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
    end#Benchmark DiagSF

    @testset "Benchmark FullSF on GPU" begin
        using SynapticFilter: _filter_update!
        using CUDA

        if CUDA.functional()
            CUDA.allowscalar(false)
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
    end#Benchmark FullSF on GPU
    end#end if
end#filters.jl
