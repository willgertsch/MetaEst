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

    # storage
    storage_q1::Vector{T}
    storage_n1::Vector{T}
    storage_n2::Vector{T}
    storage_n3::Vector{T}
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

    n, q2, q1 = size(X, 1), size(X, 2), size(Z, 2)

    yty = abs2(norm(Y))
    ztz = transpose(Z) * Z
    zty = transpose(Z) * Y

    storage_q1 = Vector{T}(undef, q1)
    storage_n1 = Vector{T}(undef, n)
    storage_n2 = Vector{T}(undef, n)
    storage_n3 = Vector{T}(undef, n)

    # return struct
    LnmObs(
        Y,
        X,
        Z,
        yty,
        ztz,
        zty,
        storage_q1,
        storage_n1,
        storage_n2,
        storage_n3
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
    logl!(obs::LnmObs, β₁, β₂, γ, σ)

Evaluate the log-likelihood of a single LNM datum at the parameter values.
Using ! notation because internal storage vectors are updated
"""
function logl!(
    obs::LnmObs{T},
    β₁::Vector{T},
    β₂::Vector{T},
    γ::Vector{T},
    σ::T
) where T <: AbstractFloat

    n = size(obs.X, 1)
    copy!(obs.storage_q1, β₁)
    obs.storage_q1 .+= β₂

    mul!(obs.storage_n1, obs.X, γ)
    mul!(obs.storage_n2, obs.Z, obs.storage_q1)
    mul!(obs.storage_n3, obs.Z, β₁)

    ll = 0
    @inbounds for i in 1:n

        πη = ilogit(obs.storage_n1[i])
        ll += log(πη * exp(-1/(2σ^2) * (obs.Y[i] - obs.storage_n2[i])^2) +
        (1 - πη) * exp(-1/(2σ^2) * (obs.Y[i] - obs.storage_n3[i])^2))
    end

    ll += -n/2 * log(2π * σ^2) 

    return ll

end


# struct to hold Lnm data and parameters
# going to update parameters, so mutable
# need to review best practices here
mutable struct LnmModel{T <: AbstractFloat}

    obs::LnmObs{T}
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

    # construct objective function
    function f(x)

        # name parameters
        β₁ = x[1:2]
        β₂ = x[3:4]
        γ = x[5:6]
        σ = x[7]

        # obs is external
        # flip sign for minimizer
        fx = -logl!(m.obs, β₁, β₂, γ, σ)
        gx = [0.0]
        hx = [0.0]

        return fx, gx, hx
    end

    # set bounds
    # need an algorithm to automatically set good bounds
    bounds = [
    0. -50. -50. 0. -10. -10. 0.;
    150. 50. 50. 50. 10. 10. 10.]

    # call optimizer
    result = optimize(
        f,
        bounds,
        method
    )

    # extract parameters
    x = result.best_sol.x
    m.β₁ = x[1:2]
    m.β₂ = x[3:4]
    m.γ = x[5:6]
    m.σ = x[7]


    # return likelihood
    return logl!(m.obs, m.β₁, m.β₂, m.γ, m.σ)
    # is this best practice?
    # return is different from object updated

end
