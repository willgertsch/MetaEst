
"""
ilogit(η::Vector{T})

Evaluate inverse-logit function for vector η.
"""
function ilogit(
η::T
) where T <: AbstractFloat
1 / (1 + exp(-η))
end

"""
ilogit!(η::Vector{T})

An in-place version of the inverse logistic function.
"""
function ilogit!(η::Vector{T}) where T <: AbstractFloat

@inbounds for i in eachindex(η)
    η[i] = ilogit(η[i])
end

end