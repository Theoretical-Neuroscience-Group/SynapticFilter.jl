struct OUModel{T1, T2}
    τ::T1 # time constant
    σs::T2 # variance
end

struct NeuronModel{T1, T2}
    g0::T1 # baseline firing rate
    β::T2 # gain
end

struct NeuronObs{T1, T2}
    x::T1 # input spikes
    y::T2 # output spikes
end
