module MetaEst

# load dependencies
using Metaheuristics
using Distributions
using LinearAlgebra
using Random

include("gaussmix.jl")
include("metamix.jl")

export GaussMixtMod, logl, GaussMixtObj, metamix

end # module
