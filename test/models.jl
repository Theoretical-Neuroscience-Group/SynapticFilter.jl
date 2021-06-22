@testset "models.jl" begin
    @testset "PoissonExpModel" begin
        @testset "Zero firing rate, zero time constant" begin
            x = ones(10)
            model = PoissonExpModel(0, 0)
            update!(x, model, 0.01)

            @test all(x .== 0)
        end

        @testset "Zero firing rate, infinite time constant" begin
            x = ones(10)
            model = PoissonExpModel(0, Inf)
            update!(x, model, 0.01)

            @test all(x .== 1)
        end

        @testset "Zero firing rate, unit time constant" begin
            x = ones(10)
            model = PoissonExpModel(0, 1)
            update!(x, model, 0.01)

            @test all(x .== exp(-0.01))
        end

        @testset "Unit firing rate, infinite time constant" begin
            x = ones(10)
            model = PoissonExpModel(1, Inf)
            update!(x, model, 1.)

            @test all(x .>= 1)
        end
    end#PoissonExpModel

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

        Smodel = OUModel(1, 0)
        Smodel = OUModel(1, 0)
        update!(w, Smodel, 0.01)

        @test all(w .== 0.99)
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
end#models.jl
