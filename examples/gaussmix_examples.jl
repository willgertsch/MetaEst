# test cases for Gaussian mixture models
using MetaEst
using Distributions
mod = GaussMixtMod([1.,2.,3.,4.], 2);
logl([.5, .5], [-1.,1], [4 /9, 4 /9], mod)
@assert abs(logl([.5, .5], [-1.,1], [4 /9, 4 /9], mod) - -38.64208) < 1e-4

# benchmark loglik function
using BenchmarkTools
using Random
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
Random.seed!(1234)
y = rand(p, 1000);
mod = GaussMixtMod(y, 2);
mu = [-1., -1.];
w = [.5, .5];
sigma = [4/9, 4/9];
@benchmark logl($w, $mu, $sigma, $mod)
# current best: 91.5ns, 0 memory alloc
# real time is more like 21.5μs

# test function factory using structs
# this is for creating functions for use with the optimizer
using MetaEst
using Distributions
using Random
using BenchmarkTools
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
Random.seed!(1234)
y = rand(p, 1000);
mod = GaussMixtMod(y, 2);
mu = [-1., 1.];
w = [.5, .5];
sigma = [4/9, 4/9];
θ = vcat(w, mu, sigma);
obj = GaussMixtObj(mod)
obj(θ)[1]

@assert obj(θ)[1] == -logl(w, mu, sigma, mod)
@benchmark obj($θ)
# this introduces allocations unfortunately

# testing Metaheuristics.jl
using MetaEst
using Distributions
using Metaheuristics
using Random

# define objective function
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
Random.seed!(1234)
y = rand(p, 1000);
mod = GaussMixtMod(y, 2);
obj = GaussMixtObj(mod);

# objective function
# annoying that I still have to wrap this
# maybe can redesign, but works fine for now
function f(x)
    return(obj(x))
end

# set bounds for variables
bounds = [0. 0. -10. -10. 0. 0.; 1. 1. 10. 10. 10. 10.]

# Common options
options = Options(seed=1234, store_convergence = true)

# set up algorithm
# real coding for GA needs special operators
ga = GA(;mutation=PolynomialMutation(;bounds),
                crossover=SBX(;bounds),
                environmental_selection=GenerationalReplacement(),
                options = options
               );

# call optimizer
result = optimize(
    f, 
    bounds, 
    DE(N = 100, options = options)
)

# visualize
using Plots
gr()

# objective function values
f_values = fvals(result);
plot(f_values)

# convergence
# need Options(store_convergence = true)
f_calls, best_f_value = convergence(result);
plot(xlabel="f calls", ylabel="fitness", title="Convergence")
plot!(f_calls, best_f_value, label="ECA")


