struct BART
    trees::Vector{Node}
end

(bart::BART)(X::Vector{Vector{Float64}}) = predictions(X, bart.trees)
Base.copy(bart::BART) = BART(copy(bart.trees))

function bart(X, y, n, M)
    # XXX standardize y
    posterior = []
    Threads.@threads for i = 1:n
        println(Threads.threadid())
        bart = BART([Leaf(0.) for _ in 1:M])
        update!(X, y, bart)
        push!(posterior, copy(bart))
    end
    posterior
end

# Gibbs update
function update!(X, y, bart)
    trees = bart.trees
    for j in eachindex(trees)
        others = [trees[i] for i in eachindex(trees) if i ≠ j]
        residuals = y .- predictions(X, others)
        trees[j] = infer_tree(X, residuals)
    end
end

predictions(X, trees) = [prediction(x, trees) for x in X]
prediction(x, trees)  = sum(tree(x) for tree in trees)
