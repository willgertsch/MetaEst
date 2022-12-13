# code for logistic normal mixture models

# define type that holds an LNMM datum
# contains m measurements over time
struct LnmmObs{T <: AbstractFloat}

    # data
    Y::Vector{T} # outcome
    X::Vector{T} # ith row latent subgroup design matrix
    Z::Matrix{T} # regression covariates

    # computed values
    yty::T
    ztz::Matrix{T}
    zty::Vector{T}

    # storage
    storage_m1::Vector{T}
    storage_m2::Vector{T}
    storage_m3::Vector{T}
    storage_mm1::Matrix{T}
end


# constructor
"""
    LnmObs(Y::Vector, X::Matrix, Z::Matrix)

Create an LNM datum of type `LnmObs`.
"""
function LnmmObs(
    Y::Vector{T},
    X::Vector{T},
    Z::Matrix{T}
) where T <: AbstractFloat

    m = size(Z, 1)

    yty = abs2(norm(Y))
    ztz = transpose(Z) * Z
    zty = transpose(Z) * Y

    # storage
    storage_m1 = Vector{T}(undef, m)
    storage_m2 = Vector{T}(undef, m)
    storage_m3 = Vector{T}(undef, m)
    storage_mm1 = Matrix{T}(undef, m, m)

    # return struct
    LnmmObs(
        Y,
        X,
        Z,
        yty,
        ztz,
        zty,
        storage_m1,
        storage_m2,
        storage_m3,
        storage_mm1
    )
end

# slower implementation for testing
function naive_logl!(
    obs::LnmmObs{T},
    β₁::Vector{T},
    β₂::Vector{T},
    γ::Vector{T},
    σ::T,
    τ::T
) where T <: AbstractFloat

    m = size(obs.Z, 1)

    p = ilogit(dot(obs.X, γ))
    Ωinv = (1/σ^2) * (I - ones(m) * ones(m)' / (σ^2 / τ^2 + m))
    r1 = obs.Y .- obs.Z * (β₁ .+ β₂) 
    r2 = r1 .+  obs.Z * β₂
    A = r1' * Ωinv * r1
    B = r2' * Ωinv * r2
    ll = -0.5m * log(2π) - m*log(σ) - 0.5*log(1 + m * τ^2 / σ^2)
    ll += log(p * exp(-0.5 * A) + (1 - p) * exp(-0.5 * B))
    return ll

end



"""
    logl!(obs::LnmmObs, β₁, β₂, γ, σ)

Evaluate the log-likelihood of a single LNMM datum at the parameter values.
Each datum is a single individual
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

    m = size(obs.Z, 1)
    # P(δᵢ = 1 | Xᵢ) = π(X'ᵢγ)
    p = ilogit(dot(obs.X, γ))

    # σ⁻²(I - (σ²/τ² + m)⁻¹ 11')
    obs.storage_mm1 .= -((σ/τ)^2 + m)
    @inbounds for i in 1:m
        obs.storage_mm1[i, i] += 1
    end
    obs.storage_mm1 ./= σ^2

    # B = (Yᵢ - Zᵢβ₁)'Ω⁻¹(Yᵢ - Zᵢβ₁)
    mul!(obs.storage_m1, obs.Z, β₁)
    mul!(obs.storage_m2, obs.storage_mm1, obs.storage_m1)
    copy!(obs.storage_m3, obs.Y)
    BLAS.gemv!('N', 1.0, obs.Z, β₁, -2.0, obs.storage_m3)
    mul!(obs.storage_m1, obs.storage_mm1, obs.Y)
    B = dot(obs.Y, obs.storage_m1) + 
    dot(obs.storage_m2, obs.storage_m3)

    # A = (Yᵢ - Zᵢ(β₁ + β₂))'Ω⁻¹(Yᵢ - Zᵢ(β₁ + β₂))
    # C = (2Yᵢ + Zᵢβ₂)'Ω⁻¹ᵢZᵢβ₂
    mul!(obs.storage_m1, obs.Z, β₂)
    mul!(obs.storage_m2, obs.storage_mm1, obs.storage_m1)
    BLAS.gemv!('N', 1.0, obs.Z, β₂, 2.0, obs.storage_m3)
    C = dot(obs.storage_m2, obs.storage_m3)

    ll = -0.5m * log(2π) - m*log(σ) - 0.5*log(1+m*(τ/σ)^2)
    # sneaky time 😎
    temp = exp(-0.5C)
    if temp == 0
        ll += -0.5B + log(p * exp(-0.5C) + 1 - p)
    elseif temp == Inf
        ll += -0.5B - 0.5C + log(p + (1 - p) * exp(0.5C))
    else
        ll += -0.5B + log(p * exp(-0.5C) + 1 - p)
    end

    return ll


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
    p = size(obsvec[1].X, 1)
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

"""
    logl!(m:LnmmModel)

Evaluate the log-likelihood for a LNMM model at parameter values
"""
function logl!(
    m::LnmmModel{T},
    β₁::Vector{T},
    β₂::Vector{T},
    γ::Vector{T},
    σ::T,
    τ::T
    ) where T <: AbstractFloat

    logl = zero(T)
    @inbounds for i in 1:length(m.data)
        obs = m.data[i]
        logl += logl!(obs, β₁, β₂, γ, σ, τ)
    end
    logl

end

"""
 fit!(m::LnmmModel, method)

 Fit a LnmmModel using a metaheuristic algorithm
 """
 function fit!(m::LnmmModel, 
    method::Metaheuristics.AbstractAlgorithm,
    bounds::Matrix{T}) where T <: AbstractFloat

    # dims
    p = size(m.data[1].X, 1)
    q = size(m.data[1].Z, 2)
    
    # objective function wrapper
    function f(x)

        # parameters
        β₁ = x[1:q]
        β₂ = x[(q+1):(2q)]
        γ = x[(2q+1):(2q+p)]
        σ = x[2q+p+1]
        τ = x[2q+p+2]

        # output functions
        fx = -logl!(m, β₁, β₂, γ, σ, τ)
        gx = [0.0]
        hx = [0.0]

        return fx, gx, hx

    end

    # bounds
    # hard coding for now
    # treat x time interaction we will be looking for an increase
    # bounds = [
    #     50.  -30. -10. -10. -20. -10. -10. 0. -5. -5. -5. 0. 0.;
    #     150. 30. 10. 10. 20. 10. 10. 10. 5. 5. 5. 10. 10.
    # ]

    # call optimizer
    result = optimize(
        f,
        bounds,
        method
    )

    # extract parameters
    x = result.best_sol.x
    m.β₁ .= x[1:q]
    m.β₂ .= x[(q+1):(2q)]
    m.γ .= x[(2q+1):(2q+p)]
    m.σ = x[2q+p+1]
    m.τ = x[2q+p+2]

    # return ll
    return logl!(m, m.β₁, m.β₂, m.γ, m.σ, m.τ)

 end