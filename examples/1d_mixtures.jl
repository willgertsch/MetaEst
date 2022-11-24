using MetaEst
using Distributions
using Random
using BenchmarkTools

# sample from mixture model
p = MixtureModel([Weibull(), Gamma()])
Random.seed!(1234)
Y = rand(p, 100)

# create mmModel
mod = mmModel(Y, p)

# update parameters of internal mixture model object
θ = [[2., 2.], [2., 2.]]
w = [.7, .3]
update_param!(mod, θ, w)

# compute log-likelihood at true values
logl!(mod, θ, w)

@benchmark logl!($mod, $θ, $w)
# 6.7μs, 3.61 KiB, 32 allocs

