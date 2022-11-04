# example and testing of 1d Gaussian mixture model
using MetaEst
using Random
using Distributions

# generate data
# case 2 from GA hydrology paper
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
Random.seed!(1234)
N = 1000
g = 2
Y = rand(p, N);
obs = GmmObs(Y, 2)


# test log-likelihood evaluation
w = [.5, .5];
μ = [-1., 1.];
σ = [4/9, 4/9];
logl!(obs, w, μ, σ)
using BenchmarkTools
@benchmark logl!($obs, $w, $μ, $σ)
# 37 μs, 0 bytes, 0 alloc


# test fitting function
mod = GmmModel(obs)
using Metaheuristics
ll = fit!(mod, DE())

mod.w
mod.μ
mod.σ
