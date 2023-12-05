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

# Inference hyper-parameters: default values.
λdefault = (n_particles=100, steps=3, n_mcmc=3)

function infer_tree(X, y; θ=θdefault, λ=λdefault)
    get_retval(last(get_traces(infer_tree_state(X, y, θ=θ, λ=λ))))
end

function infer_tree_state(X, y; θ=θdefault, λ=λdefault)
    (;n_particles,steps,n_mcmc) = λ
    σ = 0.1
    # Observation values are fixed but Gen wants fresh addresses on each update.
    obs(t) = choicemap(map(i -> (:y=>t=>i, y[i]), eachindex(y))...)
    state = pf_initialize(tree, (θ, 0, σ, X), obs(0), n_particles)
    for t in 1:steps
        pf_resize!(state, max(1, div(n_particles, t*5)))
        pf_rejuvenate!(state, mh, (CutSelection(),), n_mcmc*t)
        pf_rejuvenate!(state, mh, (LeafSelection(),), n_mcmc*t)
        argsdiff = (NoChange, UnknownChange, UnknownChange, NoChange)
        args     = (θ,        t,             σ/t,           X)
        pf_update!(state, args, argsdiff, obs(t))
    end
    return state
end

function proposal(tr, depth)

end

function nsmc(X, y; θ, λ)
    obs(t) = choicemap(map(i -> (:y=>t=>i, y[i]), eachindex(y))...)
    state = pf_initialize(tree, (θ, 0, σ, X), obs(0), n_particles)
    for depth in 1:5
        pf_resample!(state)
        pf_update!(state, args, argdiffs, observations,
                   cut_depth_proposal, (depth,))
    end
end

function cut(trace, depth)
    constraints = get_selected(get_choices(trace), CutDepthSelection(depth))
    (new_trace, w, _, discard) = update(tr, (), (), constraints)
    new_trace
end

# Select choices up to a maximum choicemap-tree depth.
struct CutDepthSelection <: Selection
    depth::Int64
end
Base.isempty(::CutDepthSelection) = false
Base.in(addr, ::CutDepthSelection) = true
Base.getindex(cut::CutDepthSelection, addr) = cut.depth > 1 ? CutDepthSelection(cut.depth-1) : EmptySelection()

# Recursive selection of all cuts.
struct CutSelection <: Selection end
Base.in(addr, ::CutSelection) = addr in [:left, :right, :cut]
Base.isempty(::CutSelection) = false # "it is not guaranteed to be empty"
Base.getindex(::CutSelection, addr) = CutSelection()

# Recursive selection of all leaf node constants.
struct LeafSelection <: Selection end
Base.in(addr, ::LeafSelection) = addr in [:left, :right, :val]
Base.isempty(::LeafSelection) = false # "it is not guaranteed to be empty"
Base.getindex(::LeafSelection, addr) = LeafSelection()

