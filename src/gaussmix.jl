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

# EM algorithm with random start for comparison
# g:number of classes
# y: data
# function em(g::Int, y::Vector{T}, maxiter::Int) where T <: AbstractFloat

#     # construct objective for evaluating logliklihood
#     mod = GaussMixtMod(y, g)

#     # random start
#     # set reasonable bounds
#     # means should be bounded by data max/mutation
#     μ_lb = minimum(y)
#     μ_ub = maximum(y)

#     # can approx each normal sd by range/4
#     # therefore should be able to bound by 0 to range/4
#     σ_ub = (maximum(y) - minimum(y))/4

#     # generate initial value uniformly in range
#     w = rand(Uniform(0, 1), g)
#     μ = rand(Uniform(μ_lb, μ_ub), g)
#     σ = rand(Uniform(0, σ_ub), g)

#     # calculate objective value
#     obj_old = logl(w, μ, σ, mod)

#     # run until we stop improving objective value
#     ftolrel = 1e-12
#     T = Matrix{Float64}(undef, mod.n, g)
#     for iter in 1:maxiter

#         # E step
#         Q = 0
#         @inbounds for i in 1:n
#             denom_i = dot(w, pdf.(Normal(μ, σ), y[i]))
#             @inbounds for j in 1:g
#                 # update posterior probabilities
#                 T[i, j] = w[j] * pdf(Normal(μ[j], σ[j]), y[i])
#             end
#         end

#         # M step
#         w = sum(T, dims = 2)./ mod.n

#         # calculate log-likelihood


#         # check monotonicity
#         obj < obj_old && (@warn "monotoniciy violated")

#         # warning about non-convergence
#         iter == maxiter && (@warn "maximum iterations reached")

#         # check convergence criterion
#         (obj - obj_old) < ftolrel * (abs(obj_old) + 1) && break


#     end

#     # return parameter estimates

# end
