module MetaEst

# load dependencies
using Metaheuristics
using Distributions
using LinearAlgebra
using Random
using GaussianMixtures # using EM algorithm from this package

include("gaussmix.jl")
include("MetaMix.jl")
include("SimMetaMix.jl")

export GaussMixtMod, 
logl, 
GaussMixtObj, 
MetaMix, 
SimTwoGaussMixEst,
RMSE,
bias

end # module
