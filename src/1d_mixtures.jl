# code for mixture models for flood frequency analysis
# 2 Gaussians
# Gamma + Weibull
# Gamma + GEV
# GEV +GEV
# all of these can be found in Distributions.jl
# therefore should just make a general interface for mixtures from Distributions.jl

# struct to hold MM data and parameters
# also include storage vectors for computation
mutable struct mmModel{T <: AbstractFloat}

    # data
    Y::Vector{T}

    # model
    model::AbstractMixtureModel

    # dims
    n::Int
    g::Int

    # parameters
    # vector of vectors to accomodate different numbers of parameters
    θ::Vector{Vector{T}}
    w::Vector{T}

    # storage
    W::Matrix{T}

end

"""
    mmModel(Y::Vector{Float}, model::AbstractMixtureModel)
Construct a general 1d mixture model. Model parameters stored in the MixtureModel
object are only temporary values used for likelihood calculation. Parameter
values should be stored in the θ and w fields.
"""
function mmModel(
    Y::Vector{T},
    model::AbstractMixtureModel
) where T <: AbstractFloat

    # dims
    n = size(Y, 1)
    g = ncomponents(model)

    # initialize parameter data structure
    # values come from values specified by user in MixtureModel object
    θ = Vector{Vector{T}}(undef, g)
    for i in 1:g
        θ[i] = Vector{T}(undef, length(params(model.components[i])))
        θ[i] .= params(components(model)[i])
    end

    w = model.prior.p

    # storage
    W = Matrix{T}(undef, n, g)

    # return
    mmModel(Y, model, n, g, θ, w, W)

end


"""
    update_param!(mod::mmModel)

Updates the parameters of a mixture model. Useful for likelihood calculation.
"""
function update_param!(
    mod::mmModel, 
    θ::Vector{Vector{T}}, 
    w::Vector{T}
    ) where T <: AbstractFloat

    # extract distribution families
    family = typeof.(mod.model.components)

    # apply new parameter values
    # should check number of arguments are correct for each
    for i in 1:mod.g
        # splatting FTW
        mod.model.components[i] = family[i](θ[i]...)
    end

    # update weights
    mod.model.prior.p .= w

end


"""
    logl!(mod::mmModel, param::Vector{Vector})

Compute log-likelihood function of mixture model at parameter values.
Using ! notation because internal storage vectors may be updated.
"""
function logl!(
    mod::mmModel{T},
    param::Vector{Vector{T}},
    w::Vector{T}
) where T <: AbstractFloat

    # construct a mixture model with given parameters


    mm = MixtureModel(distributions, w)


end
