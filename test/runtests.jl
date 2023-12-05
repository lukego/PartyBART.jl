using Test
using StatsBase
using Distributions
import PartyBART

@testset "BART" begin
    g(x) = 10sin.(pi * x[1] .* x[2]) + 20(x[3] .- 0.5).^2 + 10x[4] + 5x[5]
    n = 250
    p = 10
    σ = sqrt(1)
    X = rand(n, p)
    y = vec(mapslices(g, X, dims = 2)) + rand(Normal(0, σ), n)
    Xhat = collect(Vector{Float64}, eachrow(X))
    yhat = StatsBase.transform(fit(UnitRangeTransform, y), y)
    PartyBART.bart(Xhat, yhat, 1, 3)
end

@testset "Sine" begin
    g(x) = sin.(pi*20*x[1]) + 10*x[2]
    n = 250
    p = 10
    Xhat = Vector{Float64}[]
    for i in 0:1/n:1
        push!(Xhat, [i;1-i;rand(p-2)])
    end
    y = g.(Xhat)
    yhat = StatsBase.transform(fit(UnitRangeTransform, y), y)
    bart = PartyBART.bart(Xhat, yhat, 10, 200)
end
