using SSNVariability ; const S=SSNVariability
using Calculus, Distributions, LinearAlgebra, Statistics
using Random
using Test

Random.seed!(0)

@testset "input-output functions" begin
  g1=S.ReLu(1.2)
  @test g1(3.3) ≈ 3.3*1.2
  g2=S.ReQuad(4.4)
  @test g2(3.3) ≈ 3.3*3.3*4.4
  # gradients
  x=map( x->x+0.01*sign(x), rand(100) .- 0.2)
  for g in (g1,g2)
    @test g(-0.01) ≈ 0.0
    an = S.ioprime.(x,g)
    num = Calculus.gradient.(g,x)
    @test all( isapprox.(an,num;atol=0.01) )
  end
end

# Jacobian !
@testset "Jacobian matrix" begin
  ne,ni = 13,10
  ntot = ne+ni
  ntw = S.RecurrentNeuralNetwork(ne,ni)
  veli(u,i) = getindex(S.du_nonoise(u,ntw),i)
  @info "building the Jacobian numerically"
  utest = randn(ntot)
  Jnum = Matrix{Float64}(undef,ntot,ntot)
  for i in 1:ntot
    Jnum[i,:] =  Calculus.gradient( u -> veli(u,i), utest )
  end
  Jan=S.jacobian(utest,ntw)
  @test all(isapprox.(Jnum,Jan;rtol=1E-3))
end

@testset "2D Linear model Vs analytic results" begin

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

end

@testset "Find mean with `true` variance computed from simulation" begin
  ne,ni = 43,22
  ntot = ne+ni
  cov_noise_test=S.random_covariance_matrix(ntot,3.3)
  ntw = S.RecurrentNeuralNetwork(ne,ni;sigma_noise=cov_noise_test)

  # more weight variability
  wscal = rand(Uniform(0.5,1.5),S.n_neurons(ntw))
  broadcast!(*,ntw.weight_matrix,ntw.weight_matrix,wscal)
  mu_out,_ = S.mustart_sigmastart(ntw)
  r_sim_start = ntw.iofun.(mu_out)

  t,ev_v,ei=S.run_network_noise(ntw,r_sim_start,40.;stepsize=0.01)
  r_sim_out = median(ei;dims=2)[:]
  cov_diag_an = diag(cov(ev_v;dims=2))

  dmu_alloc = similar(mu_out)
  ε = 1E-3
  for i in 1:5000
    S.dmu!(dmu_alloc,mu_out,cov_diag_an,ntw)
    mu_out .+= ε*dmu_alloc
  end
  r_an_out = ntw.iofun.(mu_out)
  @test all(isapprox.(r_an_out,r_sim_out;atol=0.25))
end

@testset "Mean and covariance matrix with diagonal noise" begin

  Ttot = 300.0
  dt = 0.001
  ne,ni = 22,12
  ntot = ne+ni
  cov_noise_test = Symmetric(diagm(rand(Uniform(6.2,10.3),ntot)))
  ntw = S.RecurrentNeuralNetwork(ne,ni;sigma_noise=cov_noise_test)

  # more weight variability
  wscal = rand(Uniform(1.7,3.3),S.n_neurons(ntw))
  broadcast!(*,ntw.weight_matrix,ntw.weight_matrix,wscal)

  mu_an,sigma_an = S.mustart_sigmastart(ntw)
  r_start = ntw.iofun.(mu_an)
  t,ei_v,ei=S.run_network_noise(ntw,r_start,Ttot;stepsize=dt)
  sigma_num = cov(ei_v;dims=2)
  mu_num = mean(ei_v;dims=2)[:]

  dSigma_alloc = similar(sigma_num)
  dmu_alloc = similar(mu_num)

  ε = 1E-3
  for i in 1:5_000
    S.dmu!(dmu_alloc,mu_an,diag(sigma_an),ntw)
    mu_an .+= ε .* dmu_alloc
    S.dSigma!(dSigma_alloc,sigma_an,mu_an,ntw)
    sigma_an .+= ε .* dSigma_alloc
  end
  @test all(isapprox.(mu_an,mu_num;atol=0.05))
  @test all(isapprox.(sigma_an,sigma_num;atol=0.05))
end

@testset "Mean and covariance with non-diagonal noise" begin
  
  Ttot = 100.0
  dt = 0.05E-3
  ne,ni = 22,11
  ntot = ne+ni
  cov_noise_test = let c = S.random_covariance_matrix(ntot,1.0) 
    rr = Diagonal(sqrt.(rand(Uniform(2.3,8.3),ntot)))
    Symmetric(rr*c*rr)
  end
  ntw = S.RecurrentNeuralNetwork(ne,ni;sigma_noise=cov_noise_test)

  wscal = rand(Uniform(1.7,2.0),S.n_neurons(ntw))
  broadcast!(*,ntw.weight_matrix,ntw.weight_matrix,wscal)

  mu_an,sigma_an = S.mustart_sigmastart(ntw)
  r_start = ntw.iofun.(mu_an)
  t,ei_v,ei=S.run_network_noise(ntw,r_start,Ttot;stepsize=dt)
  sigma_num = cov(ei_v[:,1:100:end];dims=2)
  mu_num = mean(ei_v;dims=2)[:]

  dSigma_alloc = similar(sigma_num)
  dmu_alloc = similar(mu_num)

  ε = 0.5E-3
  for i in 1:10_000
    S.dmu!(dmu_alloc,mu_an,diag(sigma_an),ntw)
    mu_an .+= ε .* dmu_alloc
    S.dSigma!(dSigma_alloc,sigma_an,mu_an,ntw)
    sigma_an .+= ε .* dSigma_alloc
  end
  @test all(isapprox.(mu_an,mu_num;atol=0.05))
  @test all(isapprox.(sigma_an,sigma_num;atol=0.05))
end