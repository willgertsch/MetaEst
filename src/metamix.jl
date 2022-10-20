function metamix(
    data::Vector{T},
    g::Int,
    bounds::Matrix{T},
    seed::Int;
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

        options = Options(
            seed=seed, 
            store_convergence=true,
            iterations = max_iter
            )

        # select algorithm
        if alg == "ECA"
            A = ECA(N = swarm_size, options = options)
        elseif alg == "DE"
            A = DE(N = swarm_size, options = options)
        elseif alg == "PSO"
            A = PSO(N = swarm_size, options = options)
        elseif alg == "SA"
            A = SA(N = swarm_size, options = options)
        elseif alg == "WOA"
            A = WOA(N = swarm_size, options = options)
        elseif alg == "GA"
            A = GA(
                N = swarm_size,
                mutation=PolynomialMutation(;bounds),
                crossover=SBX(;bounds),
                environmental_selection=GenerationalReplacement(),
                options = options
           )
        elseif alg == "ϵDE"
            A = εDE(N = swarm_size, options = options)
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