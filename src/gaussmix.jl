# code for Gaussian mixture models

# create an object to preallocate
struct gaussmixObs{T <: AbstractFloat}
    y::Vector{T} # data
    g::Int # number of classes
    n::Int
end

# constructor
function gaussmixObs(y::Vector{T}, g::Int) where T <: AbstractFloat
    n = size(y, 1)
    # return struct
    gaussmixObs(y, g, n)
end

# log-likelihood
function gaussmix_ll(
    obs::gaussmixObs{T},
    w::Vector{T},
    μ::Vector{T},
    σ::Vector{T}
    ) where T <: AbstractFloat

    n = obs.n
    g = obs.gl
    LL = 0
    for i in 1:n
        temp = 0
        for j in 1:g
            temp += w[j] * pdf(Normal(μ[j], σ[j]), obs.y[i])
        end
        LL += log(temp)
    end
    return(LL)

end

# function factory
# input: a Gaussian mixture data object
struct loglikfun
    obs::gaussmixObs
end

# function based on struct
function(ll::loglikfun)(θ::Vector{T}) where T <: AbstractFloat
    g = ll.obs.g
    w = θ[1:g]
    μ = θ[(g+1):(2g)]
    σ = θ[(2g+1):(3g)]
    gaussmix_ll(ll.obs, w, μ, σ)
end