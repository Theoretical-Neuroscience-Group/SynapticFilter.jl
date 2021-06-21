struct FilterState{T1, T2}
    μ::T1
    Σ::T2
end


abstract type SF end # supertype of all synaptic filters

struct BlockSF{T1, T2} <: SF
    numblocks::Int
    blocksize::Int
    SModel::T1
    NModel::T2
end

struct DiagSF{T1, T2} <: SF
    dim::Int
    SModel::T1
    NModel::T2
end

struct FullSF{T1, T2} <: SF
    dim::Int
    SModel::T1
    NModel::T2
end

function update!(state, filter::SF, obs, dt)
    SModel = filter.SModel
    NModel = filter.NModel

    τ = SModel.τ
    σs = SModel.σs

    μ = state.μ
    Σ = state.Σ
    
    g0 = NModel.g0
    β = NModel.β
    
    x = obs.x
    y = obs.y

    _filter_update!(μ, Σ, τ, σs, g0, β, x, y, dt)
    return state
end

function _filter_update!(μ, Σ::AbstractArray{T, 3}, τ, σs, g0, β, x, y, dt) where T
    numblocks = size(Σ, 3)
    blocksize = size(Σ, 2)

    α = dt / τ
    α2 = 2*α

    # temporary array for storing the Σ * x for each block
    V = zeros(blocksize, numblocks)
    u = 0

    for i in 1:numblocks
        μ1 = view(μ, (i-1)*blocksize+1:i*blocksize) 
        Σ1 = view(Σ, :, :, i)
        
        x1 = view(x, (i-1)*blocksize+1:i*blocksize)

        v = view(V, :, i)

        v .= β .* Σ1 * x1
        u += β * dot(μ1, x1) + dot(v, v) / 2
    end

    for i in 1:numblocks
        μ1 = view(μ, (i-1)*blocksize+1:i*blocksize) 
        Σ1 = view(Σ, :, :, i)

        v = view(V, :, i)

        γ = g0 * dt * exp(u)
        μ1 .+= -μ1 .* α .+ v .* (y - γ)
        Σ1 .-= γ .* v * transpose(v) .+ (Σ1 .- σs) .* α2
    end
    return nothing
end

function _filter_update!(μ, Σ::AbstractVector, τ, σs, g0, β, x, y, dt)
    α = dt / τ
    α2 = 2*α

    v = β .* Σ .* x
    u = β * dot(μ, x)
    γ = g0 * dt * exp(u + dot(v, v) / 2)
    
    μ .+= - μ .* α .+ v .* (y - γ)
    Σ .+= -γ .* v .* v .- (Σ .- σs) .* α2
    return nothing
end

function _filter_update!(μ, Σ::AbstractMatrix, τ, σs, g0, β, x, y, dt)
    α = dt / τ
    α2 = 2*α
    v = β * Σ * x
    u = β * dot(μ, x)
    γ = g0 * dt * exp(u + 1/2 * dot(v, v))
    
    μ .+= -μ .* α .+ v .* (y - γ)
    Σ .-= γ .* v * transpose(v) .+ (Σ .- σs) .* α2
    return nothing
end
