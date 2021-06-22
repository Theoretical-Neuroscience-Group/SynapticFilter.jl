abstract type ErrorMeasure end

(f::ErrorMeasure)(sstate::SimulationState) = f(sstate.hstate, sstate.fstate)

function (f::ErrorMeasure)(
    imodel, smodel, omodel, filter; 
    num_timesteps, timestep, burnin::Int = 0
)
    sim = Simulation(imodel, smodel, omodel, filter)
    f(sim, num_timesteps, timestep, f, burnin)
end

function (f::ErrorMeasure)(sim::Simulation; num_timesteps, timestep, burnin::Int = 0) 
    run(sim, num_timesteps, timestep, f, burnin)
end

function run(sim, num_timesteps, timestep, errormeasure, burnin)
    if burnin >= num_timesteps
        throw(DomainError(burnin, "burnin must be smaller than total number of timesteps"))
    end
    sstate = SimulationState(sim.imodel, sim.filter)
    total_error = 0.
    for k in 1:num_timesteps
        time = k * timestep
        update!(sstate, sim, timestep, time)
        if k > burnin
            total_error += errormeasure(sstate)
        end
    end
    return total_error / (num_timesteps - burnin)
end

struct MSE <: ErrorMeasure end

ComputeMSE = MSE()

function (f::MSE)(hstate::State, fstate::FilterState)
    w = hstate.w 
    μ = fstate.μ
    f(w, μ)
end

(f::MSE)(w::AbstractArray, μ::AbstractArray) = mean((w .- μ).^2)
