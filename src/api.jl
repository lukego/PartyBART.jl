"""
    sample_bart(X, y) 

Sample one BART model conditioned on predictors `X` and response `y`.
"""
function bart(X, y)
end

function barts(X, y, n)
end

"""
    sample_value(bart, x)

Sample one value from model `bart` at `x`.
"""
predict_bart(bart, x) = bart(x)
