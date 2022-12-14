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
    W::Matrix{T}
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

    # storage
    W = Matrix{T}(undef, n, g)

    # return
    GmmModel(obs, w, μ, σ, W)

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
        w[w .<= 0] .= .00001 # fix GA issue
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

"""
    update_em!(m::GmmModel)

Perform the E and M steps of the EM algorithm.
Return the log-likelihood.
"""
function update_em!(m::GmmModel{T}) where T <: AbstractFloat

    n = size(m.obs.Y, 1)
    g = m.obs.g # preallocate this

    # E step
    # calculate posterior probabilities
    for i in 1:n

        # compute denom term
        # can maybe break this out of the loop?
        denomᵢ = 0
        for ℓ in 1:g
            denomᵢ += m.w[ℓ]/m.σ[ℓ] * exp(-1/(2m.σ[ℓ]^2) * (m.obs.Y[i] - m.μ[ℓ])^2)
        end

        denomᵢ /= sqrt(2π)

        # posterior probabilities
        for j in 1:g
            m.W[i, j] = m.w[j] * (2π * m.σ[j]^2)^(-1/2) * exp(-1/(2m.σ[j]^2)*(m.obs.Y[i] - m.μ[j])^2)
            m.W[i, j] /= denomᵢ
        end
    end

    # M step
    # column averages
    m.w .= sum(m.W, dims = 1)' # hunch that this allocates
    m.w ./= n

    mul!(m.μ, m.W', m.obs.Y)
    m.μ ./= m.w
    m.μ ./= n

    @inbounds for j in 1:g

        temp = 0
        @inbounds for i in 1:n
            temp += m.W[i, j] * (m.obs.Y[i] - m.μ[j])^2
        end

        m.σ[j] = sqrt(temp / (m.w[j] * n))

    end

    # return  log-likelihood
    ll = logl!(m.obs, m.w, m.μ, m.σ)
    return(ll)

end

"""
    fit_em!(m::GmmModel)

Fit a 1d Gaussian mixture model using EM with a random start.
Return log-likelihood
"""
function fit_em!(
    m::GmmModel;
    maxiter::Int = 10_000,
    ftolrel::AbstractFloat = 1e-12,
    prtfreq::Int = 0
)

    # random start
    # set reasonable bounds
    # means should be bounded by data max/mutation
    μ_lb = minimum(m.obs.Y)
    μ_ub = maximum(m.obs.Y)

    # can approx each normal sd by range/4
    # therefore should be able to bound by 0 to range/4
    σ_ub = (maximum(m.obs.Y) - minimum(m.obs.Y))/4

    # generate initial value uniformly in range
    g = m.obs.g
    w = rand(Uniform(0, 1), g)
    μ = rand(Uniform(μ_lb, μ_ub), g)
    σ = rand(Uniform(0, σ_ub), g)

    copy!(m.w, w)
    copy!(m.μ, μ)
    copy!(m.σ, σ)

     # initial update
     obj = update_em!(m)

     # iterations
     for iter in 0:maxiter
        obj_old = obj

        # EM update
        obj = update_em!(m)

        # print iteration number and objective value
        prtfreq > 0 && rem(iter, prtfreq) == 0 && println("iter=$iter, obj=$obj")

        # monotonicity warning
        obj < obj_old && (@warn "monotoniciy violated")

        # check convergence criteria
        (obj - obj_old) < ftolrel * (abs(obj_old) + 1) && break

        # failure to converge
        # won't print if in silent mode
        iter == maxiter && prtfreq > 0 && (@warn "maximum iterations reached")
     end
     
     # return log-likelihood
     ll = logl!(m.obs, m.w, m.μ, m.σ)
    return(ll)

end

"""
    fit_all!(m::GmmModel)

Fit 1d Gaussian mixture model with all methods.
Parameters associated with highest likelihood are saved to model object.
Returns a matrix of log-likelihoods and parameter estimates
"""
function fit_all!(m::GmmModel)

    # define bounds for GA
    μ_lb = minimum(m.obs.Y)
    μ_ub = maximum(m.obs.Y)
    σ_ub = (μ_ub - μ_lb)/4
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
    num_methods = length(metaheuristics) + 1
    lls = Vector{Float64}(undef, num_methods)
    w = Matrix{Float64}(undef, num_methods, size(m.w, 1))
    μ = Matrix{Float64}(undef, num_methods, size(m.μ, 1))
    σ = Matrix{Float64}(undef, num_methods, size(m.σ, 1))

    # fit using metaheuristics
    for i in eachindex(metaheuristics)

        lls[i] = fit!(m, metaheuristics[i])
        w[i, :] .= m.w
        μ[i, :] .= m.μ
        σ[i, :] .= m.σ

    end

    # EM algorithm takes last slot
    lls[num_methods] = fit_em!(m)
    w[num_methods, :] .= m.w
    μ[num_methods, :] .= m.μ
    σ[num_methods, :] .= m.σ

    # update model object with best values
    replace!(lls, NaN => -Inf) # PSO acting weird
    index = findmax(lls)[2]
    m.w .= w[index, :]
    m.μ .= μ[index, :]
    m.σ .= σ[index, :]

    # return
    return hcat(lls, w, μ, σ)


end


"""
    sim!(m::GmmModel, Nsim, sample_sizes)

Run a simulation study using all algorithms assuming that parameters
in m are the true values. Returns a dataframe of log-likelihoods and parameter estimates
"""
function sim(m::GmmModel, Nsim::Int, sample_sizes::Vector{Int})

    # save parameter true values
    w = m.w
    μ = m.μ
    σ = m.σ
    g = m.obs.g
    
    # initialize results storage
    simulation_results = Vector{Matrix{Float64}}(undef, Nsim * length(sample_sizes))
    for i in 1:Nsim * length(sample_sizes)
        simulation_results[i] = Matrix{Float64}(undef, 8, 3g + 2)
    end


    for i in eachindex(sample_sizes)

        N_i = sample_sizes[i]
        println("Simulating for sample size ", N_i)

        # simulate in parallel
        Threads.@threads for j in 1:Nsim

            # generate simulated data
            Y = Vector{Float64}(undef, N_i)
            # sample group membership with weights w
            group = StatsBase.sample(1:g, Weights(w), N_i)
            # generate samples
            @inbounds for ii in 1:N_i
                Y[ii] = rand(Normal(μ[group[ii]], σ[group[ii]]))
            end

            # fit all models to the data
            obs = GmmObs(Y, g)
            mod = GmmModel(obs)
            index = (i*Nsim - Nsim + j)
            simulation_results[index][:, 1:(3g+1)] .= fit_all!(mod)
            simulation_results[index][:, 3g+2] .= N_i

        end

    end

    # process data
    df = DataFrame(reduce(vcat, simulation_results), :auto)
    df.method = repeat(["ECA", "DE", "PSO", "SA", "WOA", "GA", "εDE", "EM"], Nsim*length(sample_sizes))
    select!(df, :method, :)

    return df

end