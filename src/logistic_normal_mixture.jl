# implement the logistic normal mixture model from Shen and He (2015)

# define a struct to hold LNM observation
struct LnmObs{T <: AbstractFloat}

    # data
    Y::Vector{T} # outcome
    X::Matrix{T} # latent subgroup design matrix
    Z::Matrix{T} # regression covariates

    # computed values
    yty::T
    ztz::Matrix{T}
    zty::Vector{T}
end

"""
    LnmObs(Y::Vector, X::Matrix, Z::Matrix)

Create an LNM datum of type `LnmObs`.
"""
function LnmObs(
    Y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T}
) where T <: AbstractFloat

    n, p, q = size(X, 1), size(X, 2), size(Z, 2)

    yty = abs2(norm(Y))
    ztz = transpose(Z) * Z
    zty = transpose(Z) * Y

    # return struct
    LnmObs(
        Y,
        X,
        Z,
        yty,
        ztz,
        zty
    )
end

"""
    ilogit(η::Vector{T})

Evaluate inverse-logit function for vector η.
"""
function ilogit(
    η::T
) where T <: AbstractFloat
    1 / (1 + exp(-η))
end

"""
    logl(obs::LnmObs, β₁, β₂, γ, σ)

Evaluate the log-likelihood of a single LNM datum at the parameter values.
"""
function logl(
    obs::LnmObs{T},
    β₁::Vector{T},
    β₂::Vector{T},
    γ::Vector{T},
    σ::T
) where T <: AbstractFloat

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)




end

"""
    naive_logl(obs::LnmObs, β₁, β₂, γ, σ)

Evaluate the log-likelihood of a single LNM datum at the parameter values.
Naive computational inefficient version to check for correctness
"""
function naive_logl(
    obs::LnmObs{T},
    β₁::Vector{T},
    β₂::Vector{T},
    γ::Vector{T},
    σ::T
) where T <: AbstractFloat

    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)

    ll = 0
    @inbounds for i in 1:n

        πη = ilogit(obs.X[i, :]' * γ)
        ll += log(πη * exp(-1/(2σ^2) * (obs.Y[i] - obs.Z[i, :]' * (β₁ + β₂))^2) +
        (1 - πη) * exp(-1/(2σ^2) * (obs.Y[i] - obs.Z[i, :]' * β₁)^2))
    end

    ll += -n/2 * log(2π * σ^2) 

    return ll

end


# struct to hold Lnm data and parameters
struct LnmModel{T <: AbstractFloat}

    data::LnmObs{T}
    β₁::Vector{T}
    β₂::Vector{T}
    γ::Vector{T}
    σ::T
end

"""
    LnmModel(data::Vector{LnmObs})

Create an LNM model that contains data and parameters.
"""
function LnmModel(obs::LnmObs{T}) where T <: AbstractFloat

    # dims
    p = size(obs.X, 2)
    q = size(obs.Z, 2)
    # parameters
    β₁ = Vector{T}(undef, q)
    β₂ = Vector{T}(undef, q)
    γ = Vector{T}(undef, p)
    σ = 1.

    # return model object
    LnmModel(obs, β₁, β₂, γ, σ)

end

"""
    fit!(m::LnmModel, algorithm::String)

Fit an `LnmModel` object by ML using a metaheuristic algorithm.
"""
function fit!(m::LnmModel, method::Metaheuristics.AbstractAlgorithm)


end