# test cases for Gaussian mixture models
using MetaEst
using Distributions
obs = gaussmixObs([1.,2.,3.,4.], 2);
@show gaussmix_ll(obs, [.5, .5], [-1.,1], [4 /9, 4 /9])
@assert abs(gaussmix_ll(obs, [.5, .5], [-1.,1], [4 /9, 4 /9]) - -38.64208) < 1e-4

# benchmark loglik function
using BenchmarkTools
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
y = rand(p, 1000);
obs = gaussmixObs(y, 2);
mu = [-1., -1.];
w = [.5, .5];
sigma = [4/9, 4/9];
@benchmark gaussmix_ll($obs, $w, $mu, $sigma)
# current best: 29.6μs, 0 memory alloc
# => slight improvement by adding @turbo to loop to 21.7μs but with 256 bytes

# test function factory using structs
# this is for creating functions for use with the optimizer
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
y = rand(p, 1000);
obs = gaussmixObs(y, 2);
mu = [-1., 1.];
w = [.5, .5];
sigma = [4/9, 4/9];
θ = vcat(w, mu, sigma);
loglikecase2 = MetaEst.loglikfun(obs)
loglikecase2(θ)
@benchmark loglikecase2($θ)
# this approach introduces allocations

# testing Metaheuristics.jl
using MetaEst
using Distributions
using Metaheuristics

# define objective function
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
y = rand(p, 1000);
obs = gaussmixObs(y, 2);
mu = [-1., 1.];
w = [.5, .5];
sigma = [4/9, 4/9];
θ = vcat(w, mu, sigma);
loglikecase2 = MetaEst.loglikfun(obs)

# set bounds for variables
bounds = [0. 0. -10. -10. 0. 0.; 1. 1. 10. 10. 10. 10.]

# objective function
# making another wrapper here
# can rewrite my original struct to avoid this nesting
function f(x)
    fx = -loglikecase2(x)
    # constraints
    w₁, w₂ = x[1], x[2]
    gx = [w₂ - w₁] # w₁ ≥ w₂ => w₂ - w₁ ≤ 0 
    hx = [w₁ + w₂ - 1]

    return fx, gx, hx

end

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


