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

  t,ev_v,ei=S.run_network_noise(ntw,r_sim_start,40.;verbose=true,stepsize=0.01)
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


# @testset "Unconnected mean and variance" begin
#   ne,ni = 13,10
#   ntot = ne+ni
#   ntw = S.RecurrentNeuralNetwork(ne,ni)
#   fill!(ntw.weight_matrix,0.0)
#   h_test = 3.0*randn(ntot)
#   copy!(ntw.base_input,h_test)
#   noise_test=S.random_covariance_matrix(ntot,3.3)
#   copy!(ntw.sigma_noise,noise_test)
#   mucurr=fill(3.,ntot)
#   sigmacurr = fill(0.0,ntot,ntot)
#   mu=copy(mucurr)
#   sigma=copy(sigmacurr)
#   for i in 1:100
#     S.dmuSigma_step!(mucurr,sigmacurr,mu,sigma,0.001,ntw)
#     copy!(mu,mucurr)
#     copy!(sigma,sigmacurr)
#   end
#   mu_an,sigma_an=S.mustart_sigmastart(ntw)
#   @test all(isapprox.(mucurr,mu_an;atol=0.05))
#   @test all(isapprox.(sigmacurr,sigma_an;atol=0.05))
# end
