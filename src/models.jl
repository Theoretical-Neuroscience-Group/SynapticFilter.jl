# `InputModel` subtypes are types of models for the dendritic input
abstract type InputModel end

# default InputModels are time-homogeneous
update!(x, model::InputModel, dt, t) = update!(x, model, dt)

struct PoissonExpModel{T1, T2} <: InputModel
    ρ::T1    # input firing rate
    τm::T2   # membrane time constant
    dim::Int # dimensionality
end

function update!(x::AbstractArray, model::PoissonExpModel, dt)
    τm = model.τm
    αm = dt / τm
    λ = dt * model.ρ

    for i in eachindex(x)
        @inbounds x[i] *= exp(-αm)
        @inbounds x[i] += rand(Poisson(λ)) / τm
    end
    
    return x
end

# input neurons are active in equal-sized blocks
struct BlockPoissonExpModel{T1, T2, T3} <: InputModel
    ρ::T1           # input firing rate
    τm::T2          # membrane time constant
    numblocks::Int  # number of blocks
    blocksize::Int  # size of blocks
    τblock::T3      # duration of activation of a single block
end

function update!(x::AbstractArray, model::BlockPoissonExpModel, dt, t)
    τm = model.τm
    αm = dt / τm
    λ = dt * model.ρ

    # decay for all indices
    x .*= exp(-αm)

    # compute scheduled block id (zero-based)
    block_id = floor(Int, (t / model.τblock) % model.numblocks)

    # sample new spikes only for indices within block:
    istart = block_id * model.blocksize + 1
    istop  = (block_id + 1) * model.blocksize
    for i in istart:istop
        @inbounds x[i] += rand(Poisson(λ)) / τm
    end

    return x
end


# `SynapseModel` subtypes are types of models for the evolution of synaptic weights
abstract type SynapseModel end

# default SynapseModels are time-homogeneous
update!(x, model::SynapseModel, dt, t) = update!(x, model, dt)

struct OUModel{T1, T2} <: SynapseModel
    τ::T1 # time constant for weight change
    σs::T2 # variance of weights
end

function update!(w::AbstractArray, model::OUModel, dt)
    α = dt / model.τ
    σ = sqrt(2 * dt * model.σs / model.τ) 
    
    for i in eachindex(w)
        # Ornstein-Uhlenbeck process
        @inbounds w[i] += -α * w[i] + σ * randn()
    end

    return w
end

struct State{T1, T2, T3}
    w::T1 # weights
    x::T2 # dendritic input
    η::T3 # adaptation state
end

State(w, x) = State(w, x, [0.])

function State(dim::Integer)
    w = zeros(dim)
    x = zeros(dim)
    return State(w, x)
end

State(imodel::PoissonExpModel) = State(imodel.dim)
State(imodel::BlockPoissonExpModel) = State(imodel.numblocks * imodel.blocksize)

function update!(state::State, model::InputModel, dt)
    update!(state.x, model, dt)
    return state
end

function update!(state::State, model::SynapseModel, dt)
    update!(state.w, model, dt)
    return state
end

function update!(state::State, model::InputModel, dt, t)
    update!(state.x, model, dt, t)
    return state
end

function update!(state::State, model::SynapseModel, dt, t)
    update!(state.w, model, dt, t)
    return state
end


# `OutputModel` subtypes are types of models for the output neuron
abstract type OutputModel end

# default OutputModels are time-homogeneous
update!(x, model::OutputModel, dt, t) = update!(x, model, dt)

struct NeuronObs{T1, T2}
    x::T1 # input spikes / current
    y::T2 # output spikes
end

# must return a NeuronObs
function update!(state::State, model::OutputModel, dt) end

struct ExpGainModel{T1, T2} <: OutputModel
    g0::T1 # baseline firing rate
    β::T2 # gain
end

function update!(state::State, model::ExpGainModel, dt)
    x  = state.x
    w  = state.w
    g0 = model.g0
    β  = model.β

    λ = g0 * dt * exp(β * dot(w, x))
    y = rand(Poisson(λ))

    return NeuronObs(x, y)
end

struct AdaptiveExpGainModel{T1, T2, T3, T4} <: OutputModel
    g0::T1 # baseline firing rate
    β::T2 # gain
    α::T3 # log firing rate change upon spikes
    τr::T4 # adaptation time constant
end

function update!(state::State, model::AdaptiveExpGainModel, dt)
    x  = state.x
    w  = state.w
    η  = state.η # is going to be modified
    g0 = model.g0
    β  = model.β
    α  = model.α
    τr = model.τr

    λ = g0 * dt * exp(β * dot(w, x) + η[1])
    y = rand(Poisson(λ))
    η .= η .* exp(-dt / τr) .+ α * y # exponential adaptation

    return NeuronObs(x, y)
end
