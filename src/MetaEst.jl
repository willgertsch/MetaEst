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

include("utilities.jl")
include("logistic_normal_mixture.jl")
include("1d_gaussian_mixture.jl")
include("1d_mixtures.jl")
include("lnmm.jl")

export 
logl,
LnmObs,
LnmModel,
GmmObs,
GmmModel,
LnmmObs,
LnmmModel,
fit!,
logl!,
ilogit,
update_em!,
fit_em!,
fit_all!,
sim,
mmModel,
update_param!,
mixlogpdf!,
logl2!,
ilogit!,
naive_logl!

end # module
