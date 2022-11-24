using MetaEst
using Distributions
using Random

# sample from mixture model
p = MixtureModel([Weibull(), Gamma()])
Y = rand(p, 100)

# create mmModel
mod = mmModel(Y, p)