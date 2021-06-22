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


struct MSE <: ErrorMeasure end

ComputeMSE = MSE()

function (f::MSE)(hstate::State, fstate::FilterState)
    w = hstate.w 
    μ = fstate.μ
    f(w, μ)
end

(f::MSE)(w::AbstractArray, μ::AbstractArray) = msd(w, μ)
