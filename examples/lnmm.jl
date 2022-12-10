# testing code for longitudinal extension
using MetaEst
using Random
using Distributions
using StatsBase
# generate data
# Model:
# Yᵢ = Zᵢ(β₁ + β₂δᵢ) + ξᵢ + ϵᵢ
# P(δᵢ = 1) = ilogit(Xᵢ'γ)
# ξᵢ ∼ N(0, τ²)
# ϵᵢ ∼ N(0, σ²I)



β₁ = [80., 0., 0., 0.] 
β₂ = [0., 0., 0., 1.] 
γ = [-1.39, 1.79, 0.5] # rate of Gene X in women = .6, in men = .2, when race constant
# being raceX increases the odds of having the gene
σ = 7. # residual variance
τ = 3. # random effect variance

Random.seed!(1234)
n = 1000 # number of individuals
ns = rand(1:7, n); # number of observations per individual
q₁ = length(β₁)
q₂ = length(γ) # design matrix dimensions
obsvec = Vector{LnmmObs{Float64}}(undef, n);
treat = repeat([0., 1.], inner = Int(n/2));
# sample men and women  80 20 
sex = StatsBase.sample([0., 1.], Weights([.8, .2]), n);
race = StatsBase.sample([0., 1.], Weights([.9, .1]), n);
X = hcat(ones(n), sex, race);
# sample Gene X data based on logistic regression model
geneX = rand.(Bernoulli.(ilogit.(X * γ)));

# generate data
df = reshape(zeros(3 + q₁ + q₂), 1, :);
for i in 1:n
    
    Z = Matrix{Float64}(undef, ns[i], q₁)
    Z[:, 1] .= 1. # intercept column
    # assign treatment
    Z[:, 2] .= treat[i]
    # fill in time values
    Z[:, 3] .= 0:(ns[i] - 1)
    # time-treatment interaction
    Z[:, 4] .= Z[:, 2] .* Z[:, 3]
    

    # generate Y
    Y = Vector{Float64}(undef, ns[i])
    if geneX[i] == 1
        μ = Z * (β₁ .+ β₂)
        
    else
        μ = Z * β₁
    end

    ξ = rand(Normal(0, τ^2)) .* ones(ns[i])
    ϵ = rand(MvNormal(ns[i], σ))
    Y .= μ + ξ + ϵ

    obsvec[i] = LnmmObs(Y, X[i, :], Z)

    id = Vector{Int}(undef, ns[i])
    id .= i
    df = vcat(df, 
    hcat(id, Y, Z, 
    collect(reshape(repeat(X[i,:], ns[i]), q₂, ns[i])'),
    repeat([geneX[i]], ns[i])
    ))
end

df = df[2:end, :];
df = DataFrame(df, :auto)

using DataFrames
using CSV
CSV.write("examples/longitudinal_data.csv", df)


