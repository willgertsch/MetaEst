module MetaEst

# load dependencies
using Metaheuristics
using Distributions
using LinearAlgebra
using Random
using GaussianMixtures # using EM algorithm from this package
using NaNMath

include("gaussmix.jl")
include("metamix.jl")
include("SimMetaMix.jl")
include("logistic_normal_mixture.jl")

export GaussMixtMod, 
logl, 
GaussMixtObj, 
MetaMix, 
SimTwoGaussMixEst,
RMSE,
bias,
LnmObs,
LnmModel,
fit!,
naive_logl,
ilogit

end # module
