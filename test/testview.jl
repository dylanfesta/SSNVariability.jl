push!(LOAD_PATH, abspath(@__DIR__,".."))

using SSNVariability ; const S = SSNVariability
using Test
using LinearAlgebra,Calculus,Statistics,Distributions
using Plots,NamedColors ; theme(:dark)
using Random
Random.seed!(0)

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

ne,ni = 53,31
ntot = ne+ni
cov_noise_test = Symmetric(diagm(rand(Uniform(2.2,3.3),ntot)))
ntw = S.RecurrentNeuralNetwork(ne,ni;sigma_noise=cov_noise_test,
  iofunction=S.ReLu(0.05))

# more weight variability
wscal = rand(Uniform(1.7,3.3),S.n_neurons(ntw))
broadcast!(*,ntw.weight_matrix,ntw.weight_matrix,wscal)

mu_start,sigma_out = S.mustart_sigmastart(ntw)
t,ei_v,ei=S.run_network_noise(ntw,mu_start,20.;verbose=true,stepsize=0.01)
sigma_num = cov(ei_v;dims=2)
v_sim_out = median(ei_v;dims=2)[:]
dSigma_alloc = similar(sigma_out)

global const ε = 1E-3
for i in 1:10_000
  S.dSigma!(dSigma_alloc,sigma_out,v_sim_out,ntw)
  sigma_out .+= ε*dSigma_alloc
end

plotvs(sigma_num,sigma_out)
plotvs(sigma_num, 37.22 .* sigma_out)

median(sigma_num./sigma_out ) 


##

ne,ni = 53,31
ntot = ne+ni
cov_noise_test = Symmetric(diagm(rand(Uniform(6.2,10.3),ntot)))
ntw = S.RecurrentNeuralNetwork(ne,ni;sigma_noise=cov_noise_test)

# more weight variability
wscal = rand(Uniform(1.7,3.3),S.n_neurons(ntw))
broadcast!(*,ntw.weight_matrix,ntw.weight_matrix,wscal)

mu_an,sigma_an = S.mustart_sigmastart(ntw)
t,ei_v,ei=S.run_network_noise(ntw,mu_start,20.;verbose=true,stepsize=0.01)
sigma_num = cov(ei_v;dims=2)
mu_num = mean(ei_v;dims=2)[:]

dSigma_alloc = similar(sigma_num)
dmu_alloc = similar(mu_num)

global const ε = 1E-3
for i in 1:10_000
  S.dmu!(dmu_alloc,mu_an,diag(sigma_an),ntw)
  mu_an .+= ε .* dmu_alloc
  S.dSigma!(dSigma_alloc,sigma_an,mu_an,ntw)
  sigma_an .+= ε .* dSigma_alloc
end

plotvs(mu_an, mu_num)
plotvs(sigma_an,0.06538 .* sigma_num)

mean(sigma_an ./ sigma_num)