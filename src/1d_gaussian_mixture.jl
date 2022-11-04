# code for Gaussian mixture models
# updates old code to new structure
# not extending to multivariate yet because covariance structures are harder

struct GmmObs{T <: AbstractFloat}

    # data
    Y::Vector{T}

    # number of classes
    g::Int

    # computed values

    # storage
end

"""
    GmmObs(Y:Matrix, g::Int)

Create an Gmm observation of type `GmmObs`.
"""
function GmmObs(
    Y::Vector{T},
    g::Int
) where T <: AbstractFloat

    n = size(Y, 1)

    # return
    GmmObs(Y, g)

end

"""
    logl!(obs::GmmObs, w, μ, σ)

Compute log-likelihood function of GMM data at parameter values.
Using ! notation because internal storage vectors may be updated.
"""
function logl!(
    obs::GmmObs{T},
    w::Vector{T},
    μ::Vector{T},
    σ::Vector{T}
) where T <: AbstractFloat

    n = size(obs.Y, 1)
    g = obs.g
    LL = 0
    @inbounds for i in 1:n
        temp = 0
        @inbounds for j in 1:g
            temp += w[j] * 
            exp(-1/(2σ[j]^2) * (obs.Y[i] - μ[j])^2)/σ[j]
        end
        LL += log(temp)
    end

    LL -= n/2 * log(2π)

    return(LL)
end

# struct to hold GMM data and parameters
mutable struct GmmModel{T <: AbstractFloat}
    
    obs::GmmObs{T}
    w::Vector{T}
    μ::Vector{T}
    σ::Vector{T}
end

"""
    GmmModel(obs::GmmObs)

Construct a GMM model that contains data and parameters
"""
function GmmModel(obs::GmmObs{T}) where T <: AbstractFloat


    # dims
    n = size(obs.Y, 1)
    g = obs.g

    # parameter vectors
    w = Vector{T}(undef, g)
    μ = Vector{T}(undef, g)
    σ = Vector{T}(undef, g)

    # return
    GmmModel(obs, w, μ, σ)

end

"""
    fit!(m::GmmModel, algorithm::String)

Fit a `GmmModel` object by ML using a metaheuristic algorithm.
"""
function fit!(m::GmmModel, method::Metaheuristics.AbstractAlgorithm)

    # construct objective function
    function f(x)

        g = m.obs.g
        # name parameters
        w = x[1:g]
        μ = x[(g+1):(2g)]
        σ = x[(2g+1):3g]

        # obs is external
        # flip sign for minimizer
        fx = -logl!(m.obs, w, μ, σ)
        gx = [0.0]
        hx = [sum(w) - 1.] # weights sum to 1

        return fx, gx, hx
    end
    
    # set bounds
    # mean is bounded by min/max of data
    μ_lb = minimum(m.obs.Y)
    μ_ub = maximum(m.obs.Y)

    # sds should be bounded by sd of data
    # can approximate by range/4
    σ_ub = (μ_ub - μ_lb)/4

    # construct bounds matrix
    g = m.obs.g
    bounds = [
        repeat([0.], g); 
        repeat([μ_lb], g); 
        repeat([0.], g); 
        repeat([1.], g); 
        repeat([μ_ub], g); 
        repeat([σ_ub], g)
    ]

    bounds = convert(Matrix{Float64}, transpose(reshape(bounds, :, 2)))

    # call optimizer
    result = optimize(
        f,
        bounds,
        method
    )

    # extract parameters
    x = result.best_sol.x
    m.w .= x[1:g]
    m.μ .= x[(g+1):(2g)]
    m.σ .= x[(2g+1):3g]

    # return log-likelihood
    return -minimum(result.best_sol.f)

    
end