module MetaEst

# load dependencies
using Metaheuristics
using Distributions
using LinearAlgebra

include("gaussmix.jl")

export gaussmix_ll, gaussmixObs

end # module
