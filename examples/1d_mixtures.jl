using MetaEst
using Distributions
using Random
using BenchmarkTools
using Metaheuristics
using StatsPlots

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

# classic bimodal two gaussian example
p = MixtureModel([Normal(-1, 4/9), Normal(1, 4/9)])
Random.seed!(1234)
Y = rand(p, 100)
mod = mmModel(Y, p)
fit!(mod, DE())


# examples for paper
# GA paper
# May streamflow for Colorado river
p = MixtureModel(
    [Normal(3409426., 997799.), Normal(1889160., 519555.)],
    [.75, .25]
)
Random.seed!(1206)
Y = rand(p, 2001-1966-1)
mod = mmModel(Y, p)
fit!(mod, DE())
mod.θ
mod.w

# gamma + GEV
# just make up values
p = MixtureModel(
    [Gamma(), GeneralizedExtremeValue(0., 1., 1.)],
    [.7, .3]
)
Random.seed!(1145)
Y = rand(p, 1000)
