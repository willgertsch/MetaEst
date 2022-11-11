using MetaEst
using Random
using Distributions
using StatsBase
using Metaheuristics
using DataFrames
using Gadfly


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
sex = StatsBase.sample([0., 1.], Weights([.6, .4]), N);
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


# plotting
d = DataFrame((Y=Y, treat=treat, sex=sex, geneX=geneX));
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

# fit
obs = LnmObs(Y, X, Z)
mod = LnmModel(obs)
MetaEst.fit!(
    mod,
    ECA()
)


# fit using all algorithms
lls, β₁s, β₂s, γs, σs = fit_all!(mod)
