# Metaheuristics_test.jl
using Metaheuristics

# going to follow the tutorial on the Github page to start
# solving a 10 dimensional problem

f(x) = 10length(x) + sum(x.^2 - 10cos.(2π*x))

D = 10
bounds = [-5ones(D) 5ones(D)]'

result = optimize(f, bounds);

@show minimum(result)
@show minimizer(result)

# further documentation
# examples
information = Information(f_optimum = 0.0) # known optimal value
options = Options(f_calls_limit = 9000*10, f_tol = 1e-5)
algorithm = ECA(information = information, options  = options)

result = optimize(f, bounds, algorithm)

# constrained optimization
function f(x)
    x,y = x[1], x[2]

    fx = (1-x)^2+100(y-x^2)^2
    gx = [x^2 + y^2 - 2]
    hx = [0.0] # handles equality constraints

    return fx, gx, hx
end

bounds = [-2.0 -2; 2 2]

optimize(f, bounds, ECA(N=30, K=3))

# multiobjective
bounds = [zeros(30), ones(30)]';
function f(x)
    v = 1.0 + sum(x .^ 2)
    fx1 = x[1] * v
    fx2 = (1 - sqrt(x[1])) * v
    fx = [fx1, fx2]

    # constraints
    gx = [0.0]
    hx = [0.0]

    return fx, gx, hx
end

optimize(f, bounds, NSGA2())
