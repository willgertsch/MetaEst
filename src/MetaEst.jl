module MetaEst

# load dependencies
using Metaheuristics
using Distributions
using LinearAlgebra

include("gaussmix.jl")

export GaussMixtMod, logl, GaussMixtObj

end # module
