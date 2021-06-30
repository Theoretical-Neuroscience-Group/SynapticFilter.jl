struct FilterState{T1, T2}
    μ::T1
    Σ::T2
end


abstract type SF end # supertype of all synaptic filters

struct BlockSF{T1, T2} <: SF
    numblocks::Int
    blocksize::Int
    SModel::T1
    OModel::T2
end

struct DiagSF{T1, T2} <: SF
    dim::Int
    SModel::T1
    OModel::T2
end

struct FullSF{T1, T2} <: SF
    dim::Int
    SModel::T1
    OModel::T2
end

struct GradientRule{T1, T2, T3} <: SF
    dim::Int
    SModel::T1
    OModel::T2
    η::T3 # learning rate
end

function FilterState(filter::DiagSF)
    dim = filter.dim
    μ = zeros(dim)
    Σ = zeros(dim)
    return FilterState(μ, Σ)
end

function FilterState(filter::BlockSF)
    numblocks = filter.numblocks
    blocksize = filter.blocksize
    μ = zeros(numblocks * blocksize)
    Σ = zeros(blocksize, blocksize, numblocks)
    return FilterState(μ, Σ)
end

function FilterState(filter::FullSF)
    dim = filter.dim
    μ = zeros(dim)
    Σ = zeros(dim, dim)
    return FilterState(μ, Σ)
end

function FilterState(filter::GradientRule)
    dim = filter.dim
    μ = zeros(dim)
    Σ = filter.η
    return FilterState(μ, Σ)
end

function update!(state::FilterState, filter::SF, obs::NeuronObs, dt)
    SModel = filter.SModel
    OModel = filter.OModel

    τ = SModel.τ
    σs = SModel.σs

    μ = state.μ
    Σ = state.Σ
    
    g0 = OModel.g0
    β = OModel.β
    
    x = obs.x
    y = obs.y

    _filter_update!(μ, Σ, τ, σs, g0, β, x, y, dt)
    return state
end

function expterm(μ::AbstractVector{T1}, x::AbstractVector{T2}, v::AbstractVector{T1}) where {T1, T2}
    sum = zero(T1)
    for i in eachindex(μ)
        @inbounds sum += x[i] * (μ[i] + v[i] / 2)
    end
    return sum
end

# update for diagonal filter
function _filter_update!(μ, Σ::AbstractVector, τ, σs, g0, β, x, y, dt)
    α = dt / τ
    α2 = 2*α

    v = β .* Σ .* x
    γ = g0 * dt * exp(β * expterm(μ, x, v))
    
    μ .+= -μ .* α .+ v .* (y - γ)
    Σ .-= γ .* v .* v .+ (Σ .- σs) .* α2
    return nothing
end

# update for block filter
function _filter_update!(μ, Σ::AbstractArray{T, 3}, τ, σs, g0, β, x, y, dt) where T
    numblocks = size(Σ, 3)
    blocksize = size(Σ, 2)

    α = dt / τ
    α2 = 2*α

    # temporary array for storing the Σ * x for each block
    V = zeros(blocksize, numblocks)
    u = 0.

    @inbounds for i in 1:numblocks
        μ1 = view(μ, (i-1)*blocksize+1:i*blocksize) 
        Σ1 = view(Σ, :, :, i)
        
        x1 = view(x, (i-1)*blocksize+1:i*blocksize)

        v = view(V, :, i)

        v .= β .* (Σ1 * x1)
        u += β * expterm(μ1, x1, v)
    end

    @inbounds for i in 1:numblocks
        μ1 = view(μ, (i-1)*blocksize+1:i*blocksize) 
        Σ1 = view(Σ, :, :, i)

        v = view(V, :, i)
        γ = g0 * dt * exp(u)

        for j in 1:size(Σ1, 2)
            @inbounds μ1[j] += -μ1[j] * α + v[j] * (y - γ)
            for i in 1:size(Σ1, 1)
                @inbounds Σ1[i, j] -= γ * v[i] * v[j] + (Σ1[i, j] - σs * (i==j)) * α2
            end
        end
    end
    return nothing
end

# update for full filter
function _filter_update!(μ, Σ::AbstractMatrix, τ, σs, g0, β, x, y, dt)
    α = dt / τ
    α2 = 2*α
    v = β * (Σ * x)
    γ = g0 * dt * exp(β * expterm(μ, x, v))

    # explicit loops for 2.5x speedup and less allocations
    for j in 1:size(Σ, 2)
        @inbounds μ[j] += -μ[j] * α + v[j] * (y - γ)
        for i in 1:size(Σ, 1)
            @inbounds Σ[i, j] -= γ * v[i] * v[j] + (Σ[i, j] - σs * (i==j)) * α2
        end
    end
    return nothing
end

function _filter_update!(μ, Σ::AnyCuMatrix, τ, σs, g0, β, x, y, dt)
    α = dt / τ
    α2 = 2*α
    v = β * (Σ * x)
    γ = g0 * dt * exp(β * (dot(μ, x) + dot(x, v) / 2))
    
    μ .+= -μ .* α .+ v .* (y - γ)
    view(Σ, diagind(Σ)) .-= σs
    Σ .-= γ .* v * transpose(v) .+ Σ .* α2
    view(Σ, diagind(Σ)) .+= σs
    return nothing
end

# update for gradient rule
function _filter_update!(μ, Σ::Number, τ, σs, g0, β, x, y, dt)
    v = (β * Σ) .* x
    γ = g0 * dt * exp(β * expterm(μ, x, v))
    
    μ .+= v .* (y - γ)
    return nothing
end
