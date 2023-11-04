using SSNVariability
using Documenter
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors,LaTeXStrings
using InvertedIndices
using Random

makedocs(;
    modules=[SSNVariability],
    authors="Dylan Festa",
    repo="https://github.com/dylanfesta/SSNVariability.jl/blob/{commit}{path}#L{line}",
    sitename="SSNVariability.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/SSNVariability.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/SSNVariability.jl",
)
