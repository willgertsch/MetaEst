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

@show mod.w;
@show mod.μ;
@show mod.σ;

# testing EM algorithm
copy!(mod.w, w)
copy!(mod.μ, μ)
copy!(mod.σ, σ)
@show w;
@show μ;
@show σ;
@show obj1 = update_em!(mod)
@show mod.w;
@show mod.μ;
@show mod.σ;

bm_emupdate = @benchmark update_em!($mod) setup=(
    copy!(mod.w, w);
    copy!(mod.μ, μ);
    copy!(mod.σ, σ);
)
# 97.75 μs, 80 bytes, 1 allocs

# test full algorithm
Random.seed!(1234)
@time fit_em!(mod, prtfreq = 1);

println("objective value at solution: ", update_em!(mod))
println()
println("solution values")
@show mod.w;
@show mod.μ;
@show mod.σ;

bm_em = @benchmark fit_em!($mod)

# test fit all function
Random.seed!(3124)
results = fit_all!(mod)

# simulation of case 2
using MetaEst
using Random
Y = Vector{Float64}(undef, 100)
g = 2
obs = GmmObs(Y, g)
mod = GmmModel(obs)
mod.w = [.5,.5]
mod.μ = [-1., 1.]
mod.σ = [4/9, 4/9]
Random.seed!(1234)
result = sim(mod, 1000, [30, 50, 100, 150, 200, 300]);
