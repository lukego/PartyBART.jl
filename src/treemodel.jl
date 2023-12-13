dict = Base.ImmutableDict
nobounds = dict{Int64, Tuple{Float64, Float64}}()

abstract type Node end

struct Branch <: Node
    var::Int
    cut::Real
    left::Node
    right::Node
end

struct Leaf <: Node
    val::Real
end

(node::Leaf)(_) = node.val
(node::Branch)(x) = x[node.var] < node.cut ? node.left(x) : node.right(x)

Base.show(io::IO, node::Branch) = @printf(io, "(x%d<=%.2f?%s:%s)", node.var, node.cut, node.left, node.right)
Base.show(io::IO, node::Leaf)   = @printf(io, "~%.3f", node.val)

variables(::Leaf) = []
variables(node::Branch) = [node.var; variables(node.left); variables(node.right)]

# Model with prior.

# Fixed model paramters: default values.
θdefault = (; α=0.95, β=2., σ=0.01, κ=2, M=1, D=10)

@gen function tree(θ, t, σ, Xs)
    expr = {*} ~ node(θ, σ, 0, nobounds)
    for (i, x) in enumerate(Xs)
        μ = expr(x)
        {:y=>t=>i} ~ normal(μ, σ)
    end
    expr
end

@gen function node(θ, σ, depth, bounds)
    (;α,β,σ,κ,M,D) = θ
    if (split ~ bernoulli(α*(β^-depth)))
        var ~ uniform_discrete(1, D)
        (lo,hi) = get(bounds, var, (0.,1.))
        cut ~ uniform(lo, hi)
        left  ~ node(θ, σ, depth+1, dict(bounds, var=>(lo,cut)))
        right ~ node(θ, σ, depth+1, dict(bounds, var=>(cut,hi)))
        return Branch(var, cut, left, right)
    else
        val ~ normal(0, σ/M*κ)
        return Leaf(val)
    end
end

