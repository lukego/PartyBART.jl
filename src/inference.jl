# inference.jl -- Nested SMC inference on tree models

function infer_tree(X, y, θ, λ)
    (;α,β,σ,κ,M,D) = θ
    oneof([infer_leaf(X); [infer_split(X, y, v) for v in 1:D]])
end

function infer_leaf(X, y)
    pf_initialize()
    for t in 2:n
        pf_resample!
        pf_update!(proposal() = val ~ val+ϵ)
    end
    resize!(n=1)
end

function infer_split(X, y, v)
    pf_initialize()
    for t in 2:n
        left = infer_tree(X, y)
        right = infer_tree(X, y)
        pf_resample!
        pf_update!(proposal() = cut ~ cut+ϵ)
    end
    Z = get_logml_estimate()
    resize!(n=1)
    trace[1].score + Z
end

function tree(θ, X)

end

@gen function tree(θ, t, X)
end

@gen function node(θ, depth, bounds)
    (;α,β,σ,κ,M,D) = θ
    if (split ~ bernoulli(α*(β^-depth)))
    else
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
