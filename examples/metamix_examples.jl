# test out metamix function
using MetaEst
using Distributions
using Random

# generate data
p = MixtureModel(Normal, [(-1, 4/9), (1, 4/9)], [.5,.5]);
Random.seed!(1234);
y = rand(p, 30);

bounds = [0. 0. -10. -10. 0. 0.; 1. 1. 10. 10. 10. 10.];
Random.seed!(14);
out = MetaMix(
    y,
    2,
    bounds,
    [0.],
    "ECA",
    100,
    1000
)

# results
out.w
out.μ
out.σ
out.loglik

# find seed that crashes ECA
# 14
for i in 1:1000
    Random.seed!(i)
    println(i)
    out = MetaMix(
    y,
    2,
    bounds,
    [0.],
    "ECA",
    100,
    1000
)
end