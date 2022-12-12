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
    W::Matrix{T}
    storage_n::Vector{T}

end

"""
    LnmModel(obs::LnmObs)

Create a LNM model that contains data and parameters.
"""
function LnmModel(obs::LnmObs{T}) where T <: AbstractFloat

    # dims
    n = size(obs.Y, 1)
    p = size(obs.X, 2)
    q = size(obs.Z, 2)
    # parameters
    β₁ = Vector{T}(undef, q)
    β₂ = Vector{T}(undef, q)
    γ = Vector{T}(undef, p)
    σ = 1.

    W = Matrix{T}(undef, n, 2)
    storage_n = Vector{T}(undef, n)

    # return model object
    LnmModel(obs, β₁, β₂, γ, σ, W, storage_n)

end

"""
    fit!(m::LnmModel, algorithm::String)

Fit a `LnmModel` object by ML using a metaheuristic algorithm.
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
    minY = minimum(m.obs.Y)
    maxY = maximum(m.obs.Y)
    rangeY = maxY - minY
    sdY = √(var(m.obs.Y))
    bounds = [
    minY -rangeY -rangeY 0. -10. -5. 0.;
    maxY rangeY rangeY rangeY 10. 5. sdY]

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

# model using all methods
# return matrix of log-likelihoods and parameter estimates
function fit_all!(m::LnmModel)

    # have to define bounds here for GA
    minY = minimum(m.obs.Y)
    maxY = maximum(m.obs.Y)
    rangeY = maxY - minY
    sdY = √(var(m.obs.Y))
    bounds = [
    minY -rangeY -rangeY 0. -10. -5. 0.;
    maxY rangeY rangeY rangeY 10. 5. sdY]

    # list of all metaheuristics and options
    metaheuristics = [
        ECA(),
        DE(),
        PSO(),
        SA(),
        WOA(),
        GA(
        mutation=PolynomialMutation(;bounds),
        crossover=SBX(;bounds),
        environmental_selection=GenerationalReplacement()
        ),
        εDE()
    ]

    # storage for results
    # + 1 since we also want to fit using EM
    num_methods = length(metaheuristics) + 1
    lls = Vector{Float64}(undef, num_methods)
    β₁ = Matrix{Float64}(undef, num_methods, size(m.β₁, 1))
    β₂ = Matrix{Float64}(undef, num_methods, size(m.β₂, 1))
    γ = Matrix{Float64}(undef, num_methods, size(m.γ, 1))
    σ = Vector{Float64}(undef, num_methods)

    # fit using metaheuristics
    for i in eachindex(metaheuristics)
        lls[i] = fit!(m, metaheuristics[i])

        β₁[i, :] = m.β₁
        β₂[i, :] = m.β₂
        γ[i, :] = m.γ
        σ[i] = m.σ

    end

    # EM algorithm
    # not ready yet
    lls[num_methods] = minimum(lls[1:num_methods-1])

    # update model object with best values
    index = findmax(lls)[2]
    m.β₁ .=  β₁[index, :]
    m.β₂ .= β₂[index, :]
    m.γ .= γ[index, :]
    m.σ = σ[index]

    
    # return
    return hcat(lls, β₁, β₂, γ, σ)


end

"""
    update_em!(m::LnmModel)
Perform the E and M steps of the EM algorithm.
Return the log-likelihood
"""
function update_em!(m::LnmmModel{T}) where T <: AbstractFloat

    n = size(m.obs.Y, 1)

    # E Step
    # calculate posterior probabilities
    mul!(m.storage_n, m.obs.X, m.γ)
    ilogit!(m.storage_n)
    for i in 1:n
        # compute denom term
        pᵢ = m.storage_n[i]
        denomᵢ = pᵢ * exp(-1/(2m.σ^2) * 
        (m.obs.Y[i] - dot(m.Z[i, :], m.β₁ + m.β₂))^2) +
        (1 - pᵢ) * exp(-1/(2m.σ^2) *
        (m.obs.Y[i] - dot(m.Z[i, :], m.β₁))^2)
        denomᵢ /= sqrt(2π * m.σ^2)

        # posterior probabilities
        m.W[i, 1] = pᵢ * exp(-1/(2m.σ^2) * 
        (m.obs.Y[i] - dot(m.Z[i, :], m.β₁ + m.β₂))^2) / denomᵢ
        m.W[i, 2] = (1 - pᵢ) * exp(-1/(2m.σ^2) *
        (m.obs.Y[i] - dot(m.Z[i, :], m.β₁))^2) / denomᵢ
    end

    # M step
    # update γ => use logistic regression
    # update β₁ 
    # update β₂
    # update σ => use linear regression

end