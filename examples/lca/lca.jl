# example use of latent class analysis
using MetaEst
using CSV
using DataFrames

# load carcinoma data
df = CSV.read("examples/lca/data/carcinoma.csv", DataFrame)
Y = Matrix(Float64.(df))

# create LCA object
K = repeat([2], 7)
lca = LCA(Y, K, 2)

