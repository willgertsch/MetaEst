# code for latent class analysis

# struct to hold LCA data and class analysis
struct LCA{T <: AbstractFloat}

    # data
    Y::Matrix{T}
    K::Vector{Int}

    # numbers
    J::Int # number of questions
    R::Int # number of classes
    N::Int # sample size

    # parameters
    Π::Vector{T} # probability of outcome k on variable j in class r
    P::Vector{T} # prior probabilities
end

"""
    LCA(Y::Matrix, K::Vector)

Create an LCA model object. The model expects the data to be in indicator form.
The numbers of each category should be provided.
"""
function LCA(
    Y::Matrix{T},
    K::Vector{Int},
    R::Int
) where T <: AbstractFloat

    N = size(Y, 1)
    J = length(K)
    Π = Vector{T}(undef, R * sum(K .- 1))
    P = Vector{T}(undef, R - 1)

    
    # return struct
    LCA(
        Y,
        K,
        J,
        R,
        N,
        Π,
        P
    )

end