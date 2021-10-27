push!(LOAD_PATH, abspath(@__DIR__,".."))

using SSNVariability ; const S = SSNVariability
using Test
using LinearAlgebra,Calculus,Statistics,Distributions
using Plots,NamedColors ; theme(:dark)
using Random
Random.seed!(0)

using OrdinaryDiffEq, StochasticDiffEq
# import DiffEqNoiseProcess: CorrelatedWienerProcess,WienerProcess

##

function eiplot(t::V,e::V,i::V) where V<:Vector{<:Real}
  plt=plot()
  plot!(plt,t,hcat(e,i) ; color=[:red :blue] , linewidth=3,leg=false)
end

function eiplot(t::V,e::M,i::M) where {V<:Vector,M<:Matrix{<:Real}}
  mp(mat)=mapslices(norm,mat;dims=1)[:]
  return eiplot(t,mp(e),mp(i))
end

function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end


##