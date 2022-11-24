using MetaEst
using Distributions
using Random
using BenchmarkTools
using Metaheuristics

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
# memory allocation is bad, not sure how it is this fast
# have to rewrite a lot of Distributions.jl to make this better
# sigh...


# test fitting function
fit!(mod, DE())
mod.θ
mod.w

# two gaussian example
p = MixtureModel([Normal(-1, 4/9), Normal(1, 4/9)])
Random.seed!(1234)
Y = rand(p, 100)
mod = mmModel(Y, p)
fit!(mod, DE())