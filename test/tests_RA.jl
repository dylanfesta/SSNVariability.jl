push!(LOAD_PATH, abspath(@__DIR__,".."))

using SSNVariability ; const S=SSNVariability
using Calculus, Distributions, LinearAlgebra, Statistics
using Random

# Building a network and triying a bunch of functions from the package.

Ttot = 100.0
dt = 0.1E-3
ne,ni = 1,1
ntot = ne+ni
cov_noise_test = let c = S.random_covariance_matrix(ntot,1.0) 
    rr = Diagonal(sqrt.(rand(Uniform(2.3,8.3),ntot)))
    Symmetric(rr*c*rr)
  end
ntw = S.RecurrentNeuralNetwork(ne,ni;sigma_noise=cov_noise_test)

mu_an,sigma_an = S.mustart_sigmastart(ntw)
r_start = ntw.iofun.(mu_an)
t,ei_v,ei=S.run_network_noise(ntw,r_start,Ttot;stepsize=dt)
sigma_num = cov(ei_v[:,1:100:end];dims=2)
mu_num = mean(ei_v;dims=2)[:]
dmu_var = similar(mu_num)
dsigma_var = similar(sigma_num)


S.j_thingy(mu_num,diag(sigma_num),ntw)
S.dmu!(dmu_var,mu_num,diag(sigma_num), ntw)
S.dSigma!(dsigma_var,sigma_num,mu_num, ntw)

S.dmu_step!(dmu_var, mu_num,sigma_num,dt,ntw)
S.dSigma_step_uniform_noise!(dsigma_var,sigma_num,mu_num,dt,ntw)
S.dmuSigma_step!(dmu_var,abs.(dsigma_var),mu_num,sigma_num,dt,ntw)  # dsigma_var is suposed to be strictly postivie

N = 10000
atol = 0.01
dt = atol
μ,Σ = S.run_integration!(ntw, N, atol, dt)
