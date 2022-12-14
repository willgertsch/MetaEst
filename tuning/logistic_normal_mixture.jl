# tuning for logistic normal model
# use DE to tune
# simulate 100 data sets
# use algorithm to fit model with parameters on each data set
# report median log-likelihood as objective
using Metaheuristics
using MetaEst
using StatsBase
using DataFrames
using CSV

# DE
# parameters: N, F, CR
function f(x)

    N = x[1] # pop size
    F = x[2] # step-size
    CR = x[3] # crossover rate

    # generate 100 datasets
    β₁ = [80., 0.]
    β₂ = [0., 30.]
    γ = [-1.39, 1.79]
    σ = 1.

    ll = Vector{Float64}(undef, 5)
    Threads.@threads for i in 1:5

        mod = generate_lnm(100, β₁, β₂, γ, σ)
        ll[i] = MetaEst.fit!(mod, DE(
        N = Int(round(N)) * 7,
        F = F,
        CR = CR
        )
        )

    end
    

    fx = -median(ll)
    gx = [0.0]
    hx = [0.0]

    return fx, gx, hx

end

bounds = [
    10 0.1 0.1;
    100 2.0 0.9
]

out = optimize(f, bounds, DE(options=Options(debug = true, time_limit = 3600)) )
# current best: 18, 0.88, 0.80
# 36, 0.31, 0.40

# PSO
# parameters: N, C1, C2, interia weight
function f(x)

    N = x[1] # pop size
    C1 = x[2] # learning rate 1
    C2 = x[3] # learning rate 2
    ω = x[4] # interia weight

    # generate 100 datasets
    β₁ = [80., 0.]
    β₂ = [0., 30.]
    γ = [-1.39, 1.79]
    σ = 1.

    ll = Vector{Float64}(undef, 5)
    Threads.@threads for i in 1:5

        mod = generate_lnm(100, β₁, β₂, γ, σ)
        ll[i] = MetaEst.fit!(mod, PSO(
        N = Int(round(N)) * 7,
        C1 = C1,
        C2 = C2,
        ω = ω
        )
        )

    end
    

    fx = -median(ll)
    gx = [0.0]
    hx = [0.0]

    return fx, gx, hx

end

bounds = [
    10 0.1 0.1 0.1;
    100 5.0 5.0 0.9
]

out_PSO = optimize(f, bounds, DE(options=Options(debug = true, time_limit = 3600.)) )
# 63, 2.39, 4.82, 0.30

# ECA
# parameters: step size, number of center of mass vectors
function f(x)

    N = x[1] # pop size
    η_max = x[2] # step size
    K = x[3] # number of vectors to generate center of mass

    # generate 100 datasets
    β₁ = [80., 0.]
    β₂ = [0., 30.]
    γ = [-1.39, 1.79]
    σ = 1.

    ll = Vector{Float64}(undef, 5)
    Threads.@threads for i in 1:5

        mod = generate_lnm(100, β₁, β₂, γ, σ)
        ll[i] = MetaEst.fit!(mod, ECA(
        N = Int(round(N)) * 7,
        η_max = η_max,
        K = Int(round(K))
        )
        )

    end
    

    fx = -median(ll)
    gx = [0.0]
    hx = [0.0]

    return fx, gx, hx

end

bounds = [
    10 0.1 1.0;
    100 5.0 15.0
]
out_ECA = optimize(f, bounds, DE(options=Options(debug = true, time_limit = 3600.)) )
# 283: 80, 3, 6


# compare tuned algorithm vs standard parameters
Nsims = 1000
lls = zeros(Nsims)
lls_tuned = zeros(Nsims)
β₁ = [80., 0.]
β₂ = [0., 30.]
γ = [-1.39, 1.79]
σ = 1.

# DE
Threads.@threads for i in 1:Nsims

    println("Simulation ", i, "/", Nsims)
    mod = generate_lnm(100, β₁, β₂, γ, σ)
    lls[i] = MetaEst.fit!(mod, DE())
    lls_tuned[i] = MetaEst.fit!(mod, DE(
        N = 36,
        F = 0.31,
        CR = 0.40
    ))
end

df = DataFrame(untuned = lls, tuned = lls_tuned)
CSV.write("tuning/DE.csv", df)

# PSO
lls = zeros(Nsims)
lls_tuned = zeros(Nsims)
Threads.@threads for i in 1:Nsims

    println("Simulation ", i, "/", Nsims)
    mod = generate_lnm(100, β₁, β₂, γ, σ)
    lls[i] = MetaEst.fit!(mod, PSO())
    lls_tuned[i] = MetaEst.fit!(mod, PSO(
        N = 63,
        C1 = 2.39,
        C2 = 4.82,
        ω = 0.3
    ))
end

df = DataFrame(untuned = lls, tuned = lls_tuned)
CSV.write("tuning/PSO.csv", df)