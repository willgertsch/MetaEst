module MetaEst

# load dependencies
using Metaheuristics
using Distributions
using LinearAlgebra
using Random
using GaussianMixtures # using EM algorithm from this package
using NaNMath
using StatsBase
using DataFrames

include("gaussmix.jl")
include("metamix.jl")
include("SimMetaMix.jl")
include("logistic_normal_mixture.jl")
include("1d_gaussian_mixture.jl")

export GaussMixtMod, 
logl, 
GaussMixtObj, 
MetaMix, 
SimTwoGaussMixEst,
RMSE,
bias,
LnmObs,
LnmModel,
GmmObs,
GmmModel,
fit!,
logl!,
ilogit,
update_em!,
fit_em!,
fit_all!,
sim

end # module
