# example and testing of logistic normal mixture model for subgroup identification
using MetaEst
using Random
using Distributions
using StatsBase


# generate data set
# Model:
# Yᵢ ∼ N(Zᵢ'(β₁ + β₂δᵢ), σ²)
# P(δᵢ = 1) = ilogit(Xᵢ'γ)
# parameter true values
β₁ = [80., 0.] # regression parameters baseline
β₂ = [0., 30.] # increases due to latent subgroup
γ = [-1.39, 1.79] # rate of Gene X in women = .6, in men = .2
σ = 1.

Random.seed!(1234)
N = 100;
# sample men and women with 3/2 ratio
sex = sample([0., 1.], Weights([.6, .4]), N);
X = hcat(ones(N), sex);
# sample Gene X data based on logistic regression model
geneX = rand.(Bernoulli.(ilogit.(X * γ)));

# assign half to treatment
treat = repeat([0., 1.], inner = Int(N/2));
Z = hcat(ones(N), treat);

# generate from the mixture distribution
Y = Vector{Float64}(undef, N)
for i in 1:N
    if geneX[i] == 1.
        μ = Z[i, :]' * (β₁ + β₂)
        Y[i] = rand(Normal(μ, σ))
    else
        μ = Z[i, :]' * β₁
        Y[i] = rand(Normal(μ, σ))
    end
end

using DataFrames
d = DataFrame((Y=Y, treat=treat, sex=sex, geneX=geneX));

using Gadfly
# see bimodal distributions in treated group
plot(d,
x = "Y", xgroup = "treat",
Geom.subplot_grid(Geom.density))

# what happens if we control for sex?
plot(d,
x = "Y", xgroup = "treat", color = "sex",
Geom.subplot_grid(Geom.density))
# we get more precision, but bimodal shape remains unexplained

# controlling for latent geneX
plot(d,
x = "Y", xgroup = "treat", color = "geneX",
Geom.subplot_grid(Geom.density))
# bimodel effect is fully accounted for

# we are unable to accurately estimate the treatment effect β₁₁ using standard methods
using GLM  
lm1 = fit(LinearModel, @formula(Y ~ treat), d)
# estimate is significantly biased

# adding an interaction effect solves the problem
lm2 = fit(LinearModel, @formula(Y ~ treat*geneX), d)
# not useful since geneX is unobserved


# testing functions
obs = LnmObs(Y, X, Z)
optimal = logl!(obs, β₁, β₂, γ, σ) # -167 for N = 100
@assert logl!(obs, β₁, β₂, γ, σ) > logl!(obs, [80., 20.], β₂, γ, σ)

using BenchmarkTools
@benchmark logl!($obs, $β₁, $β₂, $γ, $σ)
# 2.2 μs, 0 bytes, 0 allocs

# test using Metaheuristics
# define objective function
function f(x)

    # name parameters
    β₁ = x[1:2]
    β₂ = x[3:4]
    γ = x[5:6]
    σ = x[7]

    # obs is external
    # flip sign for minimizer
    fx = -logl!(obs, β₁, β₂, γ, σ)
    gx = [0.0]
    hx = [0.0]

    return fx, gx, hx
end

@assert f(vcat(β₁, β₂, γ, σ))[1] == -logl!(obs, β₁, β₂, γ, σ)

bounds = [
    0. -50. -50. 0. -10. -10. 0.;
    150. 50. 50. 50. 10. 10. 10.];

using Metaheuristics
result = optimize(
    f,
    bounds,
    DE()
)

result.best_sol.f
result.best_sol.x

# optimal values
# may be different, but should be close to true values
optimal
vcat(β₁, β₂, γ, σ)

# test model construction
mod = LnmModel(obs)

# test fitting function
MetaEst.fit!(
    mod,
    DE()
)

mod.β₁
mod.β₂
mod.γ
mod.σ