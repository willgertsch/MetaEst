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
    lpdfs::Vector{T}

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

    w = zeros(g)
    copy!(w, model.prior.p)

    # storage
    # posterior probabilites for EM
    W = Matrix{T}(undef, n, g)
    # log-pdfs for log-likelihood function
    lpdfs = Vector{T}(undef, n)
    

    # return
    mmModel(Y, model, n, g, θ, w, W, lpdfs)

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

    # return
    mod

end


"""
    mixlogpdf!(r::AbstractArray, d::AbstractMixtureModel, x::Vector)

This function is in the Distributions.jl code but is not accessible. Not sure
    why this wasn't included. 
"""
function mixlogpdf!(r::AbstractArray, d::MixtureModel, x)
    K = ncomponents(d)
    p = probs(d)
    n = length(r)
    Lp = Matrix{eltype(p)}(undef, n, K)
    m = fill(-Inf, n)
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0
            lpri = log(pi)
            lp_i = view(Lp, :, i)
            # only using 1d, so can skip
            lp_i .= logpdf.(component(d, i), x)

            # in the mean time, add log(prior) to lp and
            # update the maximum for each sample
            @inbounds for j = 1:n
                lp_i[j] += lpri
                if lp_i[j] > m[j]
                    m[j] = lp_i[j]
                end
            end
        end
    end

    fill!(r, 0.0)
    @inbounds for i = 1:K
        if p[i] > 0.0
            lp_i = view(Lp, :, i)
            for j = 1:n
                r[j] += exp(lp_i[j] - m[j])
            end
        end
    end

    @inbounds for j = 1:n
        r[j] = log(r[j]) + m[j]
    end
    return r
end


"""
    logl!(mod::mmModel, param::Vector{Vector})

Compute log-likelihood function of mixture model at parameter values.
Using ! notation because internals are updated.
"""
function logl!(
    mod::mmModel{T},
    θ::Vector{Vector{T}},
    w::Vector{T}
) where T <: AbstractFloat

    # update parameters of internal model
    update_param!(mod, θ, w)

    mixlogpdf!(mod.lpdfs, mod.model, mod.Y)

    return sum(mod.lpdfs)

end

"""
    fit!(m::mmModel, algorithm::String)

Fit a `mmModel` object by ML using a metaheuristic algorithm.
"""
function fit!(m::mmModel, method::Metaheuristics.AbstractAlgorithm)

    # construct objective function
    function f(x)

        g = m.g
        # name parameters
        w = x[1:g]
        w[w .<= 0] .= .00001 # fix GA issue
        θ = copy(m.θ) # take structure from model
        # fill θ sequentially
        # all parameters from component t are filled before t+1
        index = g+1
        for i in eachindex(θ)
            v = θ[i]
            for j in eachindex(v)
                v[j] = x[index]
                index += 1
            end
        end

        # obs is external
        # flip sign for minimizer
        fx = -logl!(m, θ, w)
        gx = [0.0]
        hx = [sum(w) - 1.] # weights sum to 1

        return fx, gx, hx
    end

    # set bounds
    g = m.g
    bounds = zeros(2, g + sum(length.(m.θ)))
    

    # bounds for weights
    bounds[1, 1:g] .= 0.
    bounds[2, 1:g] .= 1.
    index = g + 1

    # worthwhile to double check non-Gaussian bounds
    for family in typeof.(components(m.model))

        # manually specify for each distribution family
        # won't work in general, but good enough for this paper
        if family <: Normal

            # μ
            bounds[:, index] .= [minimum(m.Y), maximum(m.Y)]
            # σ
            bounds[:, index + 1] .= [0., √(var(m.Y))]
            index += 2

        elseif family <: Weibull
            
            # α
            bounds[:, index] .= [0., maximum(m.Y)]
            # θ
            bounds[:, index + 1] .= [0., maximum(m.Y)]
            index += 2

        elseif family <: Gamma

            # α
            bounds[:, index] .= [0., maximum(m.Y)]
            # θ
            bounds[:, index + 1] .= [0., maximum(m.Y)]
            index += 2

        elseif family <: GeneralizedExtremeValue

            # μ
            bounds[:, index] .= [minimum(m.Y), maximum(m.Y)]
            # σ
            bounds[:, index+1] .= [0., maximum(m.Y)]
            # ξ
            # can bound at 0 since we only are interested in x>0
            bounds[:, index+2] .= [0., maximum(m.Y)]
            index += 3
        end

    end

    # call optimizer
    result = optimize(
        f,
        bounds,
        method
    )

    # extract parameters
    x = result.best_sol.x
    # fill θ sequentially
    # all parameters from component t are filled before t+1
    index = g+1
    for i in eachindex(m.θ)
        for j in eachindex(m.θ[i])
            m.θ[i][j] = x[index]
            index += 1
        end
    end

    m.w .= x[1:g]

    # return log-likelihood
    return -minimum(result.best_sol.f)
end
