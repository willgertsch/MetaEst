# code for logistic normal mixture models

# define type that holds an LNMM datum
# contains m measurements over time
struct LnmmObs{T <: AbstractFloat}

    # data
    Y::Vector{T} # outcome
    X::Matrix{T} # latent subgroup design matrix
    Z::Matrix{T} # regression covariates

    # computed values
    yty::T
    ztz::Matrix{T}
    zty::Vector{T}

    # storage
    storage_m1::Vector{T}
end


# constructor
"""
    LnmObs(Y::Vector, X::Matrix, Z::Matrix)

Create an LNM datum of type `LnmObs`.
"""
function LnmmObs(
    Y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T}
) where T <: AbstractFloat

    m, q2, q1 = size(X, 1), size(X, 2), size(Z, 2)

    yty = abs2(norm(Y))
    ztz = transpose(Z) * Z
    zty = transpose(Z) * Y

    # storage
    storage_m1 = Vector{T}(undef, m)

    # return struct
    LnmmObs(
        Y,
        X,
        Z,
        yty,
        ztz,
        zty,
        storage_m1
    )
end

"""
    logit!(η::Vector{T})

An in-place version of the inverse logistic function.
"""
function ilogit!(η::Vector{T}) where T <: AbstractFloat

    @inbounds for i in eachindex(η)
        η[i] = ilogit(η[i])
    end

end

"""
    logl!(obs::LnmmObs, β₁, β₂, γ, σ)

Evaluate the log-likelihood of a single LNMM datum at the parameter values.
Using ! notation because internal storage vectors are updated
"""
function logl!(
    obs::LnmmObs{T},
    β₁::Vector{T},
    β₂::Vector{T},
    γ::Vector{T},
    σ::T,
    τ::T
) where T <: AbstractFloat

    # implement log-likelihood function
    m = size(obs.Z, 1)

    mul!(obs.storage_m1, obs.X, γ)
    p = ilogit.(obs.storage_m1)


    ll = -0.5m * log(2π) - m*log(σ) - 0.5*log(1 + m * τ^2 / σ^2)
    ll += log(p * exp(-0.5 * A) + (1 - p) * exp(-0.5 * B))
    

end

# define model object that contains multiple observations + parameters
mutable struct LnmmModel{T <: AbstractFloat}

    # data 
    data::Vector{LnmmObs{T}}
    # parameters
    β₁::Vector{T}
    β₂::Vector{T}
    γ::Vector{T}
    σ::T
    τ::T
    # preallocation
end

"""
    LnmmModel(data::Vector{LnmmObs})

Create an Lnmm Model that contains data and parameters
"""
function LnmmModel(obsvec::Vector{LnmmObs{T}}) where T <: AbstractFloat

    # dims
    p = size(obsvec[1].X, 2)
    q = size(obsvec[1].Z, 2)
    # parameters
    β₁ = Vector{T}(undef, q)
    β₂ = Vector{T}(undef, q)
    γ = Vector{T}(undef, p)
    σ = 1.
    τ = 1.

    # return model object
    LnmmModel(obsvec, β₁, β₂, γ, σ, τ)

end