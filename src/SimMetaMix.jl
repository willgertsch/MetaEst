# following the example of Shin et. al.
# we compare the performance of different algorithms

# function for computing RMSE
# θ: true value
# θ̂: estimated values
function RMSE(θ::T, θ̂::Vector{T}) where T <: AbstractFloat

    n = size(θ, 1)

    # compute MSE
    e = θ .- θ̂

    MSE = (1/n) * dot(e, e)

    # return √
    RMSE = √(MSE)
    return(RMSE)
end

# estimates bias
function bias(θ::T, θ̂::Vector{T}) where T <: AbstractFloat

    θ̄ = sum(θ̂)/length(θ̂)

    return(θ̄ - θ)
end

# compute simulation statistics for an algorithm at a given sample size
# Measures: parameter-specific RMSE and bias, median loglik
function SimTwoGaussMixEst(
    alg::String,
    sample_sizes::Vector{Int},
    θ::Vector{T},
    N::Int
) where T <: AbstractFloat

    # set up distribution to sample from
    w = θ[1:2]
    μ = θ[3:4]
    σ = θ[5:6]
    p = MixtureModel(Normal, [(μ[1], σ[1]), (μ[2], σ[2])], w);


    seed = rand(1:999)
    Random.seed!(seed)

    # check for parallelization
    if (Threads.nthreads() < 2)
        println("Surely you have more than ", Threads.nthreads(), " threads?")
    end

    # matrix to store results
    # 1 row for each sample size tested
    # columns are statistics
    results = Matrix{Float64}(undef, size(sample_sizes, 1), 13)

    # loop through all sample sample_sizes
    for j in 1:length(sample_sizes)

        println("Running simulation for sample size ", sample_sizes[j])

        # loop N times
        ŵ₁ = Vector{Float64}(undef, N)
        ŵ₂ = Vector{Float64}(undef, N)
        μ̂₁ = Vector{Float64}(undef, N)
        μ̂₂ = Vector{Float64}(undef, N)
        σ̂₁ = Vector{Float64}(undef, N)
        σ̂₂ = Vector{Float64}(undef, N)
        loglik = Vector{Float64}(undef, N)

        # parallel loop
        Threads.@threads for i in 1:N
            # generate dataset
            y = rand(p, sample_sizes[j])
    
            # set parameter bounds
            # means should be bounded by data max/mutation
            μ_lb = minimum(y)
            μ_ub = maximum(y)
    
            # can approx each normal sd by range/4
            # therefore should be able to bound by 0 to range/4
            σ_ub = (maximum(y) - minimum(y))/4
    
            bounds = [0. 0. μ_lb μ_lb 0 0; 1. 1. μ_ub μ_ub σ_ub σ_ub]
    
            #println(i)
            # run MetaMix
            out = MetaMix(
                y,
                2,
                bounds,
                [0.],
                alg,
                100,
                1000
            )
    
            # get parameter estimates
            ŵ₁[i] = out.w[1]
            ŵ₂[i] = out.w[2]
            μ̂₁[i] = out.μ[1]
            μ̂₂[i] = out.μ[2]
            σ̂₁[i] = out.σ[1]
            σ̂₂[i] = out.σ[2]
            loglik[i] = out.loglik
        end
    
        # compute statistics
        # problem: if all other parameters are the same, means can be swapped and have high scores
        results[j, :] = [
            RMSE(w[1], ŵ₁),
            RMSE(w[2], ŵ₂),
            RMSE(μ[1], μ̂₁),
            RMSE(μ[2], μ̂₂),
            RMSE(σ[1], σ̂₁),
            RMSE(σ[2], σ̂₂),
            bias(w[1], ŵ₁),
            bias(w[2], ŵ₂),
            bias(μ[1], μ̂₁),
            bias(μ[2], μ̂₂),
            bias(σ[1], σ̂₁),
            bias(σ[2], σ̂₂),
            median(loglik)
        ]
    

    end

    return results

end

