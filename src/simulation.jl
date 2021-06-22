struct Simulation{T1, T2, T3, T4}
    imodel::T1 # input model
    smodel::T2 # synapse model
    omodel::T3 # output model
    filter::T4 # filtering algorithm
end

struct SimulationState{T1, T2}
    hstate::T1 # hidden state
    fstate::T2 # filter state
end

# initialize a state for a fresh simulation based on an input model and a filtering algo
function SimulationState(imodel::InputModel, filter::SF)
    hstate = State(imodel)
    fstate = FilterState(filter)
    return SimulationState(hstate, fstate)
end

function update!(sstate::SimulationState, sim::Simulation, dt, t)
    hstate = sstate.hstate
    fstate = sstate.fstate

    imodel = sim.imodel
    smodel = sim.smodel
    omodel = sim.omodel
    filter = sim.filter

    update!(hstate, imodel, dt, t)    # sample new input
    update!(hstate, smodel, dt, t)    # update weights
    obs = update!(hstate, omodel, dt) # generate observation
    update!(fstate, filter, obs, dt)  # update filter state

    return sstate   
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
