# tuning for logistic normal model
# use DE to tune
# simulate 100 data sets
# use algorithm to fit model with parameters on each data set
# report median log-likelihood as objective
using Metaheuristics

# DE
# parameters: N, F, CR
function f(x)

    N = x[1]
    F = x[2]
    CR = x[3]

    

    return fx, gx, hx

end
