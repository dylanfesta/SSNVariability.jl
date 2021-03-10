

function nu_fun1(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  nr=Normal()
  return  @. α*(
    mu*cdf(nr,mu/sqrt(sigma_diag))+sqrt(sigma_diag)*pdf(nr,mu/sqrt(sigma_diag)))
end
function nu_fun2(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  nu1=nu_fun1(mu,sigma_diag,α)
  return @. mu * nu1 + α*sigma_diag*cdf(Normal(),mu/sqrt(sigma_diag))
end
function gamma_fun1(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  return @.  α*(mu*cdf(Normal(),mu/sqrt(sigma_diag)))
end
function gamma_fun2(mu::Vector{<:Real},sigma_diag::Vector{<:Real},α::Real)
  return 2 .* nu_fun1(mu,sigma_diag,α)
end
function other_j(mu::Vector{R},sigma_diag::Vector{R},
    ntw::RecurrentNeuralNetwork{R}) where R
  if typeof(ntw.iofun) <: ReQuad
      γ = gamma_fun2(mu,sigma_diag,ntw.iofun.α)
  elseif typeof(ntw.iofun) <: ReLu
      γ = gamma_fun1(mu,sigma_diag,ntw.iofun.α)
  end
  ret=copy(ntw.weight_matrix)
  broadcast!(*,ret,ret,transpose(γ))
  ret -= I
  return broadcast!(/,ret,ret,ntw.time_membrane)
end

function dmu!(des::V,mu::V,sigma_diag::V,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  if typeof(ntw.iofun) <: ReQuad
      ν = nu_fun2(mu,sigma_diag,ntw.iofun.α)
  elseif typeof(ntw.iofun) <: ReLu
      ν = nu_fun1(mu,sigma_diag,ntw.iofun.α)
  end
  @. des = ntw.base_input - mu
  BLAS.gemv!('N',1.,ntw.weight_matrix,ν,1.,des)
  return scalebytime!(des,ntw)
end

function dmu_step!(des::V, mu::V,Σ::Matrix{R},δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R}}
  Σdiag=diag(Σ)
  dmu!(des,mu,Σdiag,ntw)
  @. des=des*δ + mu
  return des
end
function dSigma_step!(des::M,Σ::M,μ::V,δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  Σdiag=diag(Σ)
  J=other_j(μ,Σdiag,ntw)
  lmul!(δ,J)
  J+=I
  copy!(des,ntw.sigma_noise)
  return BLAS.gemm!('N','T',1.0,J*Σ,J,δ,des)
end

function dmuSigma_step!(dest_mu::V,dest_sigma::M,μ::V,Σ::M,δ::Real,
    ntw::RecurrentNeuralNetwork{R}) where {R<:Real,V<:Vector{R},M<:Matrix{R}}
  dSigma_step!(dest_sigma,Σ,μ,δ,ntw)
  dmu_step!(dest_mu,μ,dest_sigma,δ,ntw)
  return nothing
end

## initial conditions

function mustart_sigmastart(ntw::RecurrentNeuralNetwork{R}) where R
  mustart = copy(ntw.base_input)
  Tm = diagm(-inv.(ntw.time_membrane ))
  sigma_start=lyap(Tm,ntw.sigma_noise)
  return mustart,sigma_start
end
