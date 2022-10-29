# code for Gaussian mixture models
# currently only supports 1d data

# type that includes data and model parameters
mutable struct GaussMixtMod{T <: AbstractFloat}
    y::Vector{T} # data
    g::Int # number of classes
    n::Int # size of data

    # parameters
    w::Vector{T}
    μ::Vector{T}
    σ::Vector{T}

    # statistics
    loglik::T

end

# constructor
function GaussMixtMod(
    y::Vector{T},
    g::Int
)  where T <: AbstractFloat

    n = size(y, 1)

    w = Vector{T}(undef, g)
    μ = Vector{T}(undef, g)
    σ = Vector{T}(undef, g)

    loglik = T(NaN)

    GaussMixtMod(y, g, n, w, μ, σ, loglik)
end

# log-likelihood function
# compute loglik value for parameters given data in model object
function logl(
    w::Vector{T},
    μ::Vector{T},
    σ::Vector{T},
    mod::GaussMixtMod{T}
) where T <: AbstractFloat

    n = mod.n
    g = mod.g
    LL = 0
    @inbounds for i in 1:n
        temp = 0
        @inbounds for j in 1:g
            temp += w[j] * pdf(Normal(μ[j], σ[j]), mod.y[i])
        end
        LL += log(temp)
    end

    return(LL)
end

# custom loglik functions for use in optimizer
struct GaussMixtObj
    mod::GaussMixtMod
end

function(ll::GaussMixtObj)(θ::Vector{T}) where T <: AbstractFloat

    # ECA has an issue where NaN's are generated
    # consider making a bug report
    # work-around
    replace!(θ, NaN=>0)

    g = ll.mod.g
    w = θ[1:g]
    μ = θ[(g+1):(2g)]
    σ = θ[(2g+1):(3g)]

    fx = -logl(w, μ, σ, ll.mod)
    gx = [0.0]
    hx = [sum(w) - 1]

    return fx, gx, hx
end
