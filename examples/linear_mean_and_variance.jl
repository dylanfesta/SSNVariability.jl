#=
# Mean and variance in a linear 2D system

Here I don't make use of the algorithm by Hennequin et al, but simply show how to capture
mean and variance analytically in a linear system. 

The main sources are: 

# Initialization
=#
using LinearAlgebra,Statistics,StatsBase,Distributions
using Plots,NamedColors,LaTeXStrings ; theme(:default) ; gr()

using InvertedIndices
using Random
Random.seed!(0)

using SSNVariability; const global S = SSNVariability

## #src
#=
##  Step 1: create a linear system

For simplicity I just consider a 2D model with one excitatory and one 
inhibitory unit. The results however work for any linear system. 

=#

ntw = let 
  iofun=S.IOIdentity()
    weight_matrix = [0.5 -0.5 ; 2.5 -0.1]
    time_membrane = [ 0.8 , 0.4]
    input = [10. , 6.0]
    s1,s2,ρ = 1.0,0.3,0.5
    sigma_noise = Symmetric([ s1  s1*s2*ρ ; s1*s2*ρ s2 ])
    @assert isposdef(sigma_noise)
    S.RecurrentNeuralNetwork(iofun,weight_matrix,time_membrane,input,sigma_noise)
  end
  mu_an,sigma_an = S.mean_cov_linear_ntw(ntw)

  r_start = mu_an
  dt = 10E-3
  Ttot = 1000.0
  t,_,ei=S.run_network_noise(ntw,r_start,Ttot;stepsize=dt)
  sigma_num = cov(ei;dims=2)
  mu_num = mean(ei;dims=2)[:]

  @test all(isapprox.(mu_num,mu_an;atol=0.3))
  @test all(isapprox.(sigma_num,sigma_an;atol=0.4))


## publish in documentation #src
thisfile = joinpath(splitpath(@__FILE__)[end-1:end]...) #src
using Literate; Literate.markdown(thisfile,"docs/src";documenter=true,repo_root_url="https://github.com/dylanfesta/SSNVariability.jl/blob/dev") #src
