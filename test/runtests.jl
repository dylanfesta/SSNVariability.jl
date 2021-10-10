using SSNVariability ; const S=SSNVariability
using Calculus, Distributions, LinearAlgebra, Statistics
using Test

@testset "SSNVariability.jl" begin
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

# W=0 , mean is input current
@testset "estimate of mean" begin
  ne,ni = 13,10
  ntot = ne+ni
  ntw = S.RecurrentNeuralNetwork(ne,ni)
  fill!(ntw.weight_matrix,0.0)
  h_test = 3.0*randn(ntot)
  copy!(ntw.base_input,h_test)
  noise_test=S.random_covariance_matrix(ntot,3.3)
  copy!(ntw.sigma_noise,noise_test)
  mucurr=fill(3.,ntot)
  sigmacurr = fill(0.0,ntot,ntot)
  mu=copy(mucurr)
  sigma=copy(sigmacurr)
  for i in 1:100
    S.dmuSigma_step!(mucurr,sigmacurr,mu,sigma,0.001,ntw)
    copy!(mu,mucurr)
    copy!(sigma,sigmacurr)
  end
  mu_an,sigma_an=S.mustart_sigmastart(ntw)
  @test all(isapprox.(mucurr,mu_an;atol=0.05))
  @test all(isapprox.(sigmacurr,sigma_an;atol=0.05))
end
