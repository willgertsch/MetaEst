module MetaEst

# load dependencies
using Metaheuristics
using Distributions
using LinearAlgebra
using Random
using GaussianMixtures # using EM algorithm from this package

include("gaussmix.jl")
include("metamix.jl")

export GaussMixtMod, logl, GaussMixtObj, metamix

end # module
