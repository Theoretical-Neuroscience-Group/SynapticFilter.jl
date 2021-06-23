@testset "models.jl" begin
    @testset "PoissonExpModel" begin
        @testset "Zero firing rate, infinite time constant" begin
            x = ones(10)
            model = PoissonExpModel(0, Inf, 10)
            update!(x, model, 0.01)

            @test all(x .== 1)
        end

        @testset "Zero firing rate, unit time constant" begin
            x = ones(10)
            model = PoissonExpModel(0, 1, 10)
            update!(x, model, 0.01)

            @test all(x .== exp(-0.01))
        end

        @testset "Unit firing rate, infinite time constant" begin
            x = ones(10)
            model = PoissonExpModel(1, Inf, 10)
            update!(x, model, 1.)

            @test all(x .>= 1)
        end
    end#PoissonExpModel

    @testset "BlockPoissonExpModel" begin
        @testset "" begin
            x = zeros(32)
            model = BlockPoissonExpModel(1, Inf, 8, 4, 1)

            @test_throws MethodError update!(x, model, 0.01)

            update!(x, model, 0.01, 0.)
            @test all(x[5:32] .== 0)

            update!(x, model, 0.01, 5.1)
            @test all(x[5:20] .== 0)
            @test all(x[25:32] .== 0)
        end
    end#BlockPoissonExpModel

    @testset "OUModel" begin
        @testset "Unit time constant, zero variance" begin
            w = ones(10)
            model = OUModel(1, 0)
            update!(w, model, 0.01)

            @test all(w .== 0.99)
        end

        @testset "unit time constant, unit variance" begin
            w = zeros(10)
            model = OUModel(1, 1)
            update!(w, model, 0.01)

            @test all(w .!= 0)
        end
    end#OUModel

    @testset "State" begin
        w = ones(10)
        x = ones(10)
        state = State(w, x)

        Imodel = PoissonExpModel(1, 0.1, 10)
        Smodel = OUModel(1, 1)

        wcopy = copy(w)
        update!(state, Imodel, 0.01)
        @test all(state.w .== wcopy)

        xcopy = copy(state.x)
        update!(state, Smodel, 0.01)
        @test all(state.x .== xcopy)
    end#State

    @testset "ExpGainModel" begin
        @testset "zero firing rate" begin
            w = rand(10)
            x = rand(10)

            wcopy = copy(w)
            xcopy = copy(x)

            state = State(w, x)
            model = ExpGainModel(0, 1)
            obs = update!(state, model, 0.01)

            @test all(state.w .== wcopy)
            @test all(state.x .== xcopy)
            @test all(obs.x   .== xcopy)
            @test all(obs.y   .== 0)
        end

        @testset "high firing rate" begin
            w = rand(10)
            x = rand(10)

            wcopy = copy(w)
            xcopy = copy(x)

            state = State(w, x)
            model = ExpGainModel(100, 1)
            obs = update!(state, model, 1.)

            @test all(state.w .== wcopy)
            @test all(state.x .== xcopy)
            @test all(obs.x   .== xcopy)
            @test all(obs.y   .> 0)
        end
    end#ExpGainModel

    @testset "AdaptiveExpGainModel" begin
        @testset "Infinite time constant, α = 1" begin
            w = rand(10)
            x = rand(10)
            η = [0.]

            wcopy = copy(w)
            xcopy = copy(x)
            ηcopy = copy(η)

            state = State(w, x, η)
            model = AdaptiveExpGainModel(1, 1, 1, Inf)
            obs = update!(state, model, 0.01)

            @test all(state.w .== wcopy)
            @test all(state.x .== xcopy)
            @test all(state.η .== obs.y)
            @test all(obs.x   .== xcopy)
        end

        @testset "Unit time constant, α = 0" begin
            w = rand(10)
            x = rand(10)
            η = [1.]

            wcopy = copy(w)
            xcopy = copy(x)

            state = State(w, x, η)
            model = AdaptiveExpGainModel(1, 1, 0, 1)
            obs = update!(state, model, 1.)

            @test all(state.w .== wcopy)
            @test all(state.x .== xcopy)
            @test all(state.η .== exp(-1))
            @test all(obs.x   .== xcopy)
        end
    end#AdaptiveExpGainModel

    if RUN_BENCHMARKS
    @testset "Benchmark PoissonExpModel" begin
        w = rand(1024)
        x = rand(1024)
        state = State(w, x)
        model = PoissonExpModel(1, 0.1, 10)

        println("")
        println("Benchmarking one update step for PoissonExpModel")
        display(@benchmark update!($state, $model, 0.01))
        println("")
        println("")
    end#Benchmark PoissonExpModel

    @testset "Benchmark BlockPoissonExpModel" begin
        w = rand(1024)
        x = rand(1024)
        state = State(w, x)
        model = BlockPoissonExpModel(1, 0.1, 128, 8, 1.)

        println("")
        println("Benchmarking one update step for BlockPoissonExpModel")
        display(@benchmark update!($state, $model, 0.01, 5.2))
        println("")
        println("")
    end#Benchmark BlockPoissonExpModel

    @testset "Benchmark OUModel" begin
        w = rand(1024)
        x = rand(1024)
        state = State(w, x)
        model = OUModel(1, 0.1)

        println("")
        println("Benchmarking one update step for OUModel")
        display(@benchmark update!($state, $model, 0.01))
        println("")
        println("")
    end#Benchmark OUModel

    @testset "Benchmark ExpGainModel" begin
        w = rand(1024)
        x = rand(1024)
        state = State(w, x)
        model = ExpGainModel(1, 0.1)

        println("")
        println("Benchmarking one update step for ExpGainModel")
        display(@benchmark update!($state, $model, 0.01))
        println("")
        println("")
    end#Benchmark ExpGainModel

    @testset "Benchmark AdaptiveExpGainModel" begin
        w = randn(1024)
        x = randn(1024)
        state = State(w, x)
        model = AdaptiveExpGainModel(1, 0.01, -1, 10.)

        println("")
        println("Benchmarking one update step for AdaptiveExpGainModel")
        display(@benchmark update!($state, $model, 0.01))
        println("")
        println("")
    end#Benchmark AdaptiveExpGainModel
    end#if
end#models.jl
