# code for fitting mixture models using metaheuristics
function MetaMix(
    data::Vector{T},
    g::Int,
    bounds::Matrix{T},
    parameters::Vector{T} = [0],
    alg::String = "ECA",
    swarm_size::Int = 100,
    max_iter::Int = 1000
) where T <: AbstractFloat
    
    # set up mixture mod data structure
    mod = GaussMixtMod(data, g)

    # switch between algorithm choices
    if alg == "EM"

        gm = GMM(g, data)
        w = gm.w
        μ = vec(gm.μ)
        σ = sqrt.(vec(gm.Σ))
        loglik = logl(w, μ, σ, mod)

    else # run Metaheuristics

        obj = GaussMixtObj(mod)
        function f(x)
            return(obj(x))
        end

        # options = Options(
        #     store_convergence=true,
        #     iterations = max_iter
        #     )

        # select algorithm
        if alg == "ECA"
            A = ECA()
        elseif alg == "DE"
            A = DE()
        elseif alg == "PSO"
            A = PSO()
        elseif alg == "SA"
            A = SA()
        elseif alg == "WOA"
            A = WOA()
        elseif alg == "GA"
            A = GA(
                mutation=PolynomialMutation(;bounds),
                crossover=SBX(;bounds),
                environmental_selection=GenerationalReplacement()
           )
        elseif alg == "ϵDE"
            A = εDE()
        else
            println("Algorithm not supported.")
        end

        # call optimizer
        result = optimize(
            f,
            bounds,
            A
        )

        # extract values from optimizer results
        sol = result.best_sol.x
        w = sol[1:g]
        μ = sol[g+1:2g]
        σ = sol[2g+1:3g]
        loglik = -result.best_sol.f
    end

    # update stored parameters and statistics
    mod.w = w
    mod.μ = μ
    mod.σ = σ
    mod.loglik = loglik

    # return gaussmixmod object
    return(mod)
end